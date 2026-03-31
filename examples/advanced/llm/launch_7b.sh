#!/bin/bash
# ── Launch 7B comparison and generate plots when done ────────────────────────
#
# Usage (from SLURM login node):
#   cd /scratch/$USER/iterativennsimple
#   bash examples/advanced/llm/launch_7b.sh
#
# This submits the 5-job array and a dependent plot-generation job.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd /scratch/$USER/iterativennsimple
mkdir -p results logs checkpoints/7b

echo "Submitting 7B job array (5 variants, 1 GPU each)..."
ARRAY_ID=$(sbatch --parsable examples/advanced/llm/llama_7b.sbatch)
echo "  Array job ID: $ARRAY_ID"
echo "  Monitor: squeue -u $USER | grep $ARRAY_ID"

# Submit plot generation job that runs after all array tasks complete
echo "Submitting plot generation job (depends on $ARRAY_ID)..."
PLOT_ID=$(sbatch --parsable \
    --dependency=afterany:$ARRAY_ID \
    --partition=short \
    --time=00:30:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --job-name=llama-7b-plots \
    --output=logs/llama_7b_plots_%j.out \
    --error=logs/llama_7b_plots_%j.err \
    --wrap "export PATH=\$HOME/.local/bin:\$PATH; cd /scratch/\$USER/iterativennsimple; uv run --extra llm examples/advanced/llm/llama_comparison.py --plot-from results/llama_7b_103.jsonl --plot-dir results/plots_7b")
echo "  Plot job ID: $PLOT_ID (runs after all training completes)"

echo ""
echo "All submitted! Commands:"
echo "  squeue -u $USER                          # check status"
echo "  tail -f logs/llama_7b_*_${ARRAY_ID}.log  # watch training"
echo "  scancel $ARRAY_ID                        # cancel all training"
echo "  scancel $PLOT_ID                         # cancel plot job"
