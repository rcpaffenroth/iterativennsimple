# Task 1-5 — Assemble the Final Notebook

## Goal

Combine all cells from Tasks 1-1 through 1-4 into a single Jupyter notebook file at:

```
notebooks/advanced/11-rcp-load-sequence-example.ipynb
```

## Instructions

1. Create a new Jupyter notebook (`.ipynb` format).
2. Add cells in order from Tasks 1-1, 1-2, 1-3, and 1-4.
3. Alternate between markdown cells (explanations) and code cells (implementations) as specified in each task.
4. Ensure all markdown uses proper LaTeX formatting for math equations (use `$$..$$` for display math).

## Cell Order (complete list)

| # | Type | Source | Content |
|---|------|--------|---------|
| 1 | markdown | Task 1-1, Cell 1 | Title and introduction |
| 2 | code | Task 1-1, Cell 2 | Imports |
| 3 | markdown | Task 1-1, Cell 3 | Data loading explanation |
| 4 | code | Task 1-1, Cell 4 | Load and split data |
| 5 | code | Task 1-1, Cell 5 | Train/val split and DataLoaders |
| 6 | markdown | Task 1-1, Cell 6 | Hyperparameters explanation |
| 7 | code | Task 1-1, Cell 7 | Hyperparameters |
| 8 | markdown | Task 1-2, Cell 1 | RNN/LSTM/GRU section header |
| 9 | code | Task 1-2, Cell 2 | SimpleRecurrentClassifier class |
| 10 | markdown | Task 1-2, Cell 3 | Training loop explanation |
| 11 | code | Task 1-2, Cell 4 | train_model function |
| 12 | markdown | Task 1-2, Cell 5 | Train the models header |
| 13 | code | Task 1-2, Cell 6 | Train RNN, LSTM, GRU |
| 14 | markdown | Task 1-3, Cell 1 | Sequential2D explanation with math |
| 15 | code | Task 1-3, Cell 2 | make_monarch_block helper |
| 16 | code | Task 1-3, Cell 3 | build_seq2d_map function |
| 17 | markdown | Task 1-3, Cell 4 | Sequence classifier explanation |
| 18 | code | Task 1-3, Cell 5 | MonarchSequenceClassifier class |
| 19 | code | Task 1-3, Cell 6 | Train Monarch (1 iteration) |
| 20 | markdown | Task 1-4, Cell 1 | Iterated Sequential2D explanation |
| 21 | code | Task 1-4, Cell 2 | Train with 2 and 3 iterations |
| 22 | markdown | Task 1-4, Cell 3 | Comparison header |
| 23 | code | Task 1-4, Cell 4 | Summary table |
| 24 | code | Task 1-4, Cell 5 | Plot validation accuracy curves |
| 25 | markdown | Task 1-4, Cell 6 | Observations |

## Notebook Format

The `.ipynb` file is JSON. Each cell is an object in the `"cells"` array:

```json
{
 "cell_type": "markdown" or "code",
 "metadata": {},
 "source": ["line 1\n", "line 2\n", "last line"],
 "outputs": [],
 "execution_count": null
}
```

- `source` is a list of strings, each ending with `\n` except the last.
- Code cells need `"outputs": []` and `"execution_count": null`.
- Markdown cells do not need `outputs` or `execution_count`.

The notebook metadata should be:

```json
{
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "name": "python",
  "version": "3.10.0"
 }
}
```

## Verification

- The file should be valid JSON and loadable by JupyterLab.
- Running all cells in order should execute without errors (assuming `generatedata` and `iterativennsimple` are installed).
- The notebook should contain approximately 25 cells.
- All markdown math should render correctly in JupyterLab.
