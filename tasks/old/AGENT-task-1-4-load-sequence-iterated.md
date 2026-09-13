# Task 1-4 — Iterated Sequential2D and Comparison

## Goal

Create the fourth section of the notebook. Demonstrate applying the Sequential2D map **multiple times per timestep** (iterations) and compare all models.

## Prerequisites

All variables from Tasks 1-1 through 1-3 are available:
- `MonarchSequenceClassifier`, `train_model`, `results` dict
- `input_dim`, `label_dim`, `hidden_size`, `num_epochs`, `learning_rate`, `device`
- `train_loader`, `val_loader`

## Background: Multiple Iterations

In Task 1-3, the Sequential2D map was applied **once** per timestep. The key insight of iterative neural networks is that we can apply the **same map multiple times** per timestep, allowing the hidden state and predictions to refine iteratively.

For 3 iterations at each timestep, the update is equivalent to:

$$\begin{bmatrix} I & 0 & 0 \\ M_1 & M_2 & M_3 \\ M_4 & M_5 & M_6 \end{bmatrix}^3 \begin{bmatrix} x_k \\ y_k \\ h_k \end{bmatrix} = \begin{bmatrix} x_{k+1} \\ y_{k+1} \\ h_{k+1} \end{bmatrix}$$

This is like applying the same matrix three times in sequence — the **weights are shared** across iterations (no extra parameters!), but the model gets more compute per timestep.

## Cells to Create

### Cell 1 — Markdown: Section header

```markdown
## 4. Iterated Sequential2D

The Sequential2D map can be applied **multiple times** per timestep. This gives the model more compute to refine its hidden state and predictions, without adding any extra parameters (the same weights are reused each iteration).

For 3 iterations at timestep $k$:

$$\underbrace{\begin{bmatrix} I & 0 & 0 \\ M_1 & M_2 & M_3 \\ M_4 & M_5 & M_6 \end{bmatrix} \begin{bmatrix} I & 0 & 0 \\ M_1 & M_2 & M_3 \\ M_4 & M_5 & M_6 \end{bmatrix} \begin{bmatrix} I & 0 & 0 \\ M_1 & M_2 & M_3 \\ M_4 & M_5 & M_6 \end{bmatrix}}_{\text{3 iterations, same weights}} \begin{bmatrix} x_k \\ y_k \\ h_k \end{bmatrix} = \begin{bmatrix} x_{k+1} \\ y_{k+1} \\ h_{k+1} \end{bmatrix}$$

Let's train models with 1, 2, and 3 iterations and see how accuracy changes.
```

### Cell 2 — Code: Train with different iteration counts

```python
for n_iters in [2, 3]:
    print(f"\n{'='*50}")
    print(f"Training MonarchLinear Sequential2D ({n_iters} iterations per timestep)")
    print(f"{'='*50}")

    # Create a new model (same architecture, fresh weights)
    model = MonarchSequenceClassifier(input_dim, label_dim, hidden_size, num_blocks=4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} (same as 1-iteration model!)")

    # We need a custom training loop that passes `iterations` to forward()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        correct, total = 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch, iterations=n_iters)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.detach().argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        train_acc = correct / total * 100

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch, iterations=n_iters)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        val_acc = correct / total * 100

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch:2d}/{num_epochs}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

    results[f"Monarch ({n_iters} iter)"] = history
    print(f"\n  Best val_acc = {max(history['val_acc']):.1f}%")
```

### Cell 3 — Markdown: Comparison

```markdown
## 5. Model Comparison

Let's compare all models side by side: the three recurrent baselines (RNN, LSTM, GRU) and the MonarchLinear models with different iteration counts.
```

### Cell 4 — Code: Summary table

```python
import matplotlib.pyplot as plt

print(f"\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}")
print(f"{'Model':<22} {'Best Val Acc':>12} {'Final Val Acc':>14}")
print("-" * 50)
for name, hist in results.items():
    best = max(hist["val_acc"])
    final = hist["val_acc"][-1]
    print(f"{name:<22} {best:>11.1f}% {final:>13.1f}%")
```

### Cell 5 — Code: Plot validation accuracy curves

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for name, hist in results.items():
    ax.plot(range(1, num_epochs + 1), hist["val_acc"], label=name, marker="o", markersize=3)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Model Comparison: Validation Accuracy over Training")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Cell 6 — Markdown: Observations

```markdown
## Observations

Key takeaways from this comparison:

1. **RNN vs LSTM vs GRU**: LSTM and GRU typically outperform vanilla RNN on sequence tasks due to their gating mechanisms that help with long-range dependencies.

2. **MonarchLinear iterations**: Increasing the number of iterations per timestep generally improves accuracy **without adding any parameters**. The same weights are applied repeatedly, giving the model more compute to refine its state.

3. **Parameter efficiency**: The MonarchLinear models use sparse Monarch matrices, which have fewer parameters than a dense linear layer of the same size. This makes them memory-efficient while still being effective.

4. **Iterative refinement**: The iterative approach (applying the same map multiple times) is fundamentally different from making the network deeper — it's more like giving the model time to "think" at each step.
```

## Verification

- Models with more iterations should generally achieve similar or better accuracy than fewer iterations (though not guaranteed with only 10 epochs).
- The parameter count should be identical across all Monarch models (1, 2, 3 iterations) since the same weights are reused.
- The plot should show all models' learning curves on one graph.
- The summary table should list all models with their best and final validation accuracies.
