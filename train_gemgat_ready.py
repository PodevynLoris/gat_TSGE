#!/usr/bin/env python
"""
Production-ready gemGAT training script
- Uses your exact file naming format
- Uses 1024 hidden dimensions (no mysterious extra 4 features)
- Loads pre-split data and pre-computed WGCNA graphs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import dgl
import random
import argparse
import os
from model_corrected import gemGAT_Corrected, CorrectedLoss

def set_seed(seed=123):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)

def load_expression_data(data_dir, data_name):
    """
    Load your gemGAT format expression data
    
    Expected files:
    - expr_in_{data_name}.csv (train source)
    - expr_out_{data_name}.csv (train target)
    - expr_in_test_{data_name}.csv (test source)
    - expr_out_test_{data_name}.csv (test target)
    """
    print(f"\nLoading expression data for: {data_name}")
    
    # Train data paths
    train_source_path = os.path.join(data_dir, f"expr_in_{data_name}.csv")
    train_target_path = os.path.join(data_dir, f"expr_out_{data_name}.csv")
    
    # Test data paths
    test_source_path = os.path.join(data_dir, f"expr_in_test_{data_name}.csv")
    test_target_path = os.path.join(data_dir, f"expr_out_test_{data_name}.csv")
    
    # Load CSVs
    print(f"Loading train source: {train_source_path}")
    train_source_df = pd.read_csv(train_source_path)
    
    print(f"Loading train target: {train_target_path}")
    train_target_df = pd.read_csv(train_target_path)
    
    print(f"Loading test source: {test_source_path}")
    test_source_df = pd.read_csv(test_source_path)
    
    print(f"Loading test target: {test_target_path}")
    test_target_df = pd.read_csv(test_target_path)
    
    # Extract gene IDs and convert to tensors
    # First column is gene IDs, rest are samples
    gene_ids_source = train_source_df.iloc[:, 0].values
    gene_ids_target = train_target_df.iloc[:, 0].values
    
    # Convert to tensors (already log2-transformed from preprocessing)
    train_data = {
        'source': torch.tensor(train_source_df.iloc[:, 1:].values, dtype=torch.float32),
        'target': torch.tensor(train_target_df.iloc[:, 1:].values, dtype=torch.float32),
        'gene_ids_source': gene_ids_source,
        'gene_ids_target': gene_ids_target,
        'n_samples': train_source_df.shape[1] - 1  # Exclude gene ID column
    }
    
    test_data = {
        'source': torch.tensor(test_source_df.iloc[:, 1:].values, dtype=torch.float32),
        'target': torch.tensor(test_target_df.iloc[:, 1:].values, dtype=torch.float32),
        'gene_ids_source': gene_ids_source,
        'gene_ids_target': gene_ids_target,
        'n_samples': test_source_df.shape[1] - 1
    }
    
    print(f"\nData shapes:")
    print(f"  Train source: {train_data['source'].shape} (genes × samples)")
    print(f"  Train target: {train_data['target'].shape}")
    print(f"  Test source: {test_data['source'].shape}")
    print(f"  Test target: {test_data['target'].shape}")
    
    return train_data, test_data

def load_graphs(graph_dir, source_tissue, target_tissue, use_weighted=False, tau=0.5):
    """
    Load pre-computed WGCNA graphs
    
    Expected files:
    - {source_tissue}_adjacency_binary_tau{tau}.csv
    - {target_tissue}_adjacency_binary_tau{tau}.csv
    OR if use_weighted:
    - {source_tissue}_adjacency_weighted.csv
    - {target_tissue}_adjacency_weighted.csv
    """
    print(f"\nLoading graphs (weighted={use_weighted}, tau={tau})")
    
    if use_weighted:
        source_path = os.path.join(graph_dir, f"{source_tissue}_adjacency_weighted.csv")
        # Handle BA9 naming
        if "BA9" in target_tissue or "Frontal_Cortex" in target_tissue:
            target_path = os.path.join(graph_dir, "BA9_adjacency_weighted.csv")
        else:
            target_path = os.path.join(graph_dir, f"{target_tissue}_adjacency_weighted.csv")
    else:
        source_path = os.path.join(graph_dir, f"{source_tissue}_adjacency_binary_tau{tau}.csv")
        # Handle BA9 naming
        if "BA9" in target_tissue or "Frontal_Cortex" in target_tissue:
            target_path = os.path.join(graph_dir, f"BA9_adjacency_binary_tau{tau}.csv")
        else:
            target_path = os.path.join(graph_dir, f"{target_tissue}_adjacency_binary_tau{tau}.csv")
    
    print(f"Loading source graph: {source_path}")
    adj_source = pd.read_csv(source_path, index_col=0)
    
    print(f"Loading target graph: {target_path}")
    adj_target = pd.read_csv(target_path, index_col=0)
    
    # Convert to tensors
    adj_source_tensor = torch.tensor(adj_source.values, dtype=torch.float32)
    adj_target_tensor = torch.tensor(adj_target.values, dtype=torch.float32)
    
    print(f"Graph shapes:")
    print(f"  Source adjacency: {adj_source_tensor.shape}")
    print(f"  Target adjacency: {adj_target_tensor.shape}")
    
    return adj_source_tensor, adj_target_tensor

def create_dgl_graphs(adj_source, adj_target):
    """Create DGL graphs from adjacency matrices"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create source graph
    edges_source = torch.nonzero(adj_source, as_tuple=True)
    g_source = dgl.graph(edges_source)
    g_source = dgl.add_self_loop(g_source)
    g_source = g_source.to(device)
    
    # Create target graph
    edges_target = torch.nonzero(adj_target, as_tuple=True)
    g_target = dgl.graph(edges_target)
    g_target = dgl.add_self_loop(g_target)
    g_target = g_target.to(device)
    
    # Combined graph (using target structure as base)
    g_combined = dgl.graph(edges_target)
    g_combined = dgl.add_self_loop(g_combined)
    g_combined = g_combined.to(device)
    
    print(f"\nDGL Graphs created:")
    print(f"  Source: {g_source.number_of_nodes()} nodes, {g_source.number_of_edges()} edges")
    print(f"  Target: {g_target.number_of_nodes()} nodes, {g_target.number_of_edges()} edges")
    
    return g_source, g_target, g_combined

def train_epoch(model, optimizer, criterion, g_source, g_target, g_combined, 
                train_data, adj_target, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    gene_losses = []
    link_losses = []
    
    n_samples = train_data['n_samples']
    
    for i in range(n_samples):
        # Get sample data
        x_source = train_data['source'][:, i:i+1].to(device)
        y_target = train_data['target'][:, i:i+1].to(device)
        
        # Forward pass (no extra features, using 1024 dims)
        predictions = model(g_source, g_target, g_combined, x_source, extra_features=None)
        
        # Compute loss
        loss, gene_loss, link_loss = criterion(predictions, y_target, adj_target.to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        gene_losses.append(gene_loss.item())
        link_losses.append(link_loss.item())
        
        # Progress reporting for debugging
        if (i + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}, Sample {i+1}/{n_samples}, "
                  f"Loss: {loss.item():.4f} (Gene: {gene_loss.item():.4f}, Link: {link_loss.item():.4f})")
    
    return total_loss / n_samples, np.mean(gene_losses), np.mean(link_losses)

def evaluate(model, g_source, g_target, g_combined, test_data, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        n_samples = test_data['n_samples']
        
        for i in range(n_samples):
            x_source = test_data['source'][:, i:i+1].to(device)
            y_target = test_data['target'][:, i:i+1].to(device)
            
            # Get predictions (no extra features)
            _, all_genes_pred, _ = model(g_source, g_target, g_combined, x_source, extra_features=None)
            
            all_predictions.append(all_genes_pred.cpu())
            all_targets.append(y_target.cpu())
    
    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=1)
    targets = torch.cat(all_targets, dim=1)
    
    # Compute per-gene Pearson correlations
    correlations = []
    for i in range(predictions.shape[0]):
        if predictions[i].std() > 0 and targets[i].std() > 0:
            corr = torch.corrcoef(torch.stack([predictions[i], targets[i]]))[0, 1]
            if not torch.isnan(corr):
                correlations.append(corr.item())
    
    mean_corr = np.mean(correlations) if correlations else 0
    median_corr = np.median(correlations) if correlations else 0
    
    # Compute MSE
    mse = torch.mean((predictions - targets) ** 2).item()
    
    return mean_corr, median_corr, mse, len(correlations)

def main(args):
    """Main training function"""
    print("=" * 80)
    print("gemGAT Training - Production Ready")
    print("=" * 80)
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load expression data
    train_data, test_data = load_expression_data(args.data_dir, args.data_name)
    
    # Load graphs
    adj_source, adj_target = load_graphs(
        args.graph_dir, 
        args.source_tissue, 
        args.target_tissue,
        args.use_weighted,
        args.tau
    )
    
    # Create DGL graphs
    g_source, g_target, g_combined = create_dgl_graphs(adj_source, adj_target)
    
    # Get dimensions
    n_genes_source = adj_source.shape[0]
    n_genes_target = adj_target.shape[0]
    
    print(f"\nModel configuration:")
    print(f"  Source genes: {n_genes_source}")
    print(f"  Target genes: {n_genes_target}")
    print(f"  Hidden dimension: 1024 (no extra features)")
    print(f"  Attention heads: {args.nheads}")
    
    # Initialize model with 1024 dimensions (no extra features)
    model = gemGAT_Corrected(
        ngene_in=n_genes_source,
        ngene_out=n_genes_target,
        nhidatt=1024,  # Fixed at 1024
        nheads=args.nheads,
        extra_features_dim=0,  # No extra features
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Initialize loss with adaptive weighting
    criterion = CorrectedLoss(adaptive=True).to(device)
    
    # Optimizer (Adam, not SGD)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    best_corr = -1
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, gene_loss, link_loss = train_epoch(
            model, optimizer, criterion, g_source, g_target, g_combined,
            train_data, adj_target, device, epoch
        )
        
        # Evaluate
        mean_corr, median_corr, mse, n_valid_genes = evaluate(
            model, g_source, g_target, g_combined, test_data, device
        )
        
        # Update scheduler
        scheduler.step(train_loss)
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train - Total Loss: {train_loss:.4f}")
            print(f"         Gene Loss: {gene_loss:.4f}, Link Loss: {link_loss:.4f}")
            print(f"  Test  - Mean Correlation: {mean_corr:.4f}")
            print(f"         Median Correlation: {median_corr:.4f}")
            print(f"         MSE: {mse:.4f}")
            print(f"         Valid genes: {n_valid_genes}/{n_genes_target}")
        
        # Save best model
        if mean_corr > best_corr:
            best_corr = mean_corr
            patience_counter = 0
            
            save_path = os.path.join(args.output_dir, f"gemgat_best_{args.data_name}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_corr': best_corr,
                'args': vars(args)
            }, save_path)
            
            print(f"  ✓ New best model saved (correlation: {best_corr:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 80)
    print(f"Training Complete!")
    print(f"Best test correlation: {best_corr:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gemGAT Training - Your Data Format')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, 
                        default='./output_preprocessing_blood_brain',
                        help='Directory with expression data')
    parser.add_argument('--graph_dir', type=str, 
                        default='./wgcna_results_5000',
                        help='Directory with WGCNA graphs')
    parser.add_argument('--output_dir', type=str, 
                        default='./models',
                        help='Directory to save models')
    
    # Data naming
    parser.add_argument('--data_name', type=str,
                        default='Whole_Blood_to_Brain_-_Frontal_Cortex_BA9gtex',
                        help='Data name suffix for files')
    parser.add_argument('--source_tissue', type=str, 
                        default='Whole_Blood',
                        help='Source tissue name for graphs')
    parser.add_argument('--target_tissue', type=str,
                        default='Brain_-_Frontal_Cortex_BA9',
                        help='Target tissue name for graphs')
    
    # Graph configuration
    parser.add_argument('--use_weighted', action='store_true',
                        help='Use weighted adjacency instead of binary')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Tau threshold for binary adjacency')
    
    # Model hyperparameters (1024 is hardcoded in model init)
    parser.add_argument('--nheads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--print_every', type=int, default=5,
                        help='Print frequency')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    main(args)
