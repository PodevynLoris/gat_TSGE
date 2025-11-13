# gemGAT Code Diagnostic Report

## Executive Summary
After analyzing the gemGAT paper, supplementary materials, and implementation code, I've identified **multiple critical errors** that explain why the loss doesn't converge. The implementation has fundamental mismatches with the paper's architecture and contains several logical bugs.

---

## Critical Issues Found

### 1. **Architecture Mismatch - Incorrect MLP Input Dimensions**

#### Paper Specification (from Supplementary):
- First MLP layer after GAT encoding: **1028 → 512**
- This suggests GAT output dimension is 1028 (likely 1024 + 4 extra features)

#### Code Implementation:
```python
# In model.py, line 44-54
self.pred_in = nn.Sequential(
    nn.Linear(self.nhidatt, 512),  # nhidatt = 1024, not 1028!
    ...
)
```

**Issue**: The paper clearly states the input should be 1028, not 1024. The authors likely concatenated additional features that are missing in the code.

---

### 2. **Graph Construction Error - Self-loops Added Incorrectly**

#### Code Issue (train.py, lines 45-48):
```python
adj = adj + torch.eye(adj.shape[0])
adj_ms = adj_ms + torch.eye(adj_ms.shape[0])
```

**Problem**: Self-loops are added to the adjacency matrix BEFORE converting to DGL graph. However, DGL graphs automatically handle self-loops internally when needed. This results in:
- Duplicate self-loops
- Incorrect edge weights
- Distorted attention mechanism

---

### 3. **Loss Function Weight Imbalance**

#### Code Issue (train.py, line 121):
```python
loss = loss_graph + 100*loss_gene
```

**Problem**: The weight of 100 for gene loss is arbitrary and likely too high. This creates:
- Extreme gradient imbalance
- Link prediction task gets essentially ignored
- Model can't learn proper graph structure

---

### 4. **Incorrect Graph Updates During Training**

#### Code Issue (train.py, lines 139-150):
```python
if (epoch + 1) % 20 == 0:
    A_semi1[A_semi1 <= 0.99998] = 0
    A_semi1[A_semi1 > 0.99998] = 1
    # ...
    adj_all.add_edges(A_r2, A_c2)
```

**Problems**:
1. Hard thresholding at 0.99998 is too aggressive
2. Only updating graph every 20 epochs disrupts gradient flow
3. Adding edges without removing old ones creates duplicate edges
4. No mechanism to remove incorrect edges

---

### 5. **Semi-supervised Link Prediction Logic Error**

#### Code Issue (model.py, lines 102-106):
```python
zsemi_lp1 = zsemi_lp[:self.ngene_out, :]
zsemi_lp2 = zsemi_lp[self.ngene_out:, :]
A_semi_ori = F.sigmoid(torch.matmul(zsemi_lp1, torch.transpose(zsemi_lp1, 0, 1)))
A_semi1 = F.sigmoid(torch.matmul(zsemi_lp1, torch.transpose(zsemi_lp2, 0, 1)))
A_semi2 = F.sigmoid(torch.matmul(zsemi_lp2, torch.transpose(zsemi_lp2, 0, 1)))
```

**Problem**: The model predicts links between:
- Source genes with themselves (A_semi_ori)
- Source genes with target genes (A_semi1)  
- Target genes with themselves (A_semi2)

But A_semi1 is never used in the loss! Only A_semi_ori is compared against the known adjacency matrix.

---

### 6. **Input Feature Processing Error**

#### Code Issue (train.py, lines 79-84):
```python
node_feat_train = torch.log(node_feat_train + 1)
node_feat_test = torch.log(node_feat_test + 1)
gene_pred_train = torch.log(gene_pred_train + 1)
gene_pred_test = torch.log(gene_pred_test + 1)
```

**Problem**: Log transformation is applied, but the model outputs are never inverse-transformed. This means:
- Model learns in log-space
- Loss is computed in log-space
- But evaluation might expect original space

---

### 7. **Missing Graph Attention Aggregation**

#### Code Issue (model.py):
The GAT outputs are reshaped but never properly aggregated:
```python
zpred = self.attentions(g1, z)
zpred = zpred.view(zpred.shape[0], -1)  # Just flattening!
```

**Problem**: Multi-head attention outputs should be:
1. Either concatenated (standard GAT)
2. Or averaged (alternative approach)

The code just reshapes without proper aggregation strategy.

---

### 8. **Optimizer and Learning Rate Issues**

#### Code Issue (train.py):
```python
optimizer = optim.SGD(model.parameters(), lr=args.lr)
lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[3, 5, 10, 15, 20, 30, 40, 50, 80, 100, 300, 500, 700, 800, 1000],
    gamma=0.1)
```

**Problems**:
1. SGD without momentum for graph neural networks is suboptimal
2. Learning rate decay by 0.1 is too aggressive
3. Milestones don't match the 300 epoch default training

---

### 9. **Data Leakage in Graph Construction**

The code uses the same adjacency matrices for both training and testing:
```python
adj = pd.read_csv(f"./data/graph_in_{args.data}.csv", header=0)
adj_ms = pd.read_csv(f"./data/graph_out_{args.data}.csv", header=0)
```

These graphs are used for ALL samples, which means:
- Test set information leaks into training through shared graph structure
- Model can memorize patterns rather than generalize

---

### 10. **Incorrect Loss Normalization**

#### Code Issue (train.py, lines 86-87):
```python
pos_weight = float(adj_semi.shape[0] * adj_semi.shape[0] - adj_semi.sum()) / adj_semi.sum()
norm = adj_semi.shape[0] * adj_semi.shape[0] / float((adj_semi.shape[0] * adj_semi.shape[0] - adj_semi.sum()) * 2)
```

**Problem**: The normalization assumes a fully connected graph, but the actual graph is sparse. This leads to:
- Incorrect positive/negative balance
- Wrong gradient scaling

---

## Recommended Fixes

### Priority 1 (Critical):
1. **Fix MLP input dimensions**: Check what the extra 4 features should be
2. **Remove duplicate self-loops**: Let DGL handle self-loops internally
3. **Fix loss weighting**: Use learnable or adaptive weights
4. **Include all link prediction losses**: Use A_semi1 in the loss computation

### Priority 2 (Important):
5. **Fix graph update mechanism**: Update continuously or use soft adjacency
6. **Switch optimizer**: Use Adam or AdamW instead of SGD
7. **Fix multi-head aggregation**: Properly concatenate or average heads
8. **Add inverse log transform**: For proper evaluation

### Priority 3 (Improvements):
9. **Separate train/test graphs**: Or use proper masking
10. **Fix loss normalization**: Account for actual graph sparsity

---

## Reproduction Recommendations

Given these issues, the paper's results are likely not reproducible with the provided code. To properly implement gemGAT:

1. **Contact the authors** for the correct implementation or missing components
2. **Re-implement from scratch** following the paper's mathematical formulations
3. **Use established GAT libraries** (PyTorch Geometric, DGL) with proper patterns
4. **Start with simpler baselines** and gradually add complexity

---

## Conclusion

The code has fundamental implementation errors that prevent proper training. The loss convergence issue is caused by multiple compounding problems:
- Architectural mismatches with the paper
- Incorrect graph operations
- Missing loss components
- Poor optimization settings

These issues explain why you're seeing non-converging loss. The code appears to be an incomplete or incorrectly modified version of the actual implementation used in the paper.
