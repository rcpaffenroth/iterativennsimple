#!/bin/bash
#SBATCH --job-name=llama-7b-rank
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000B|H200"
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/llama_7b_rank_%A_%a.out
#SBATCH --error=logs/llama_7b_rank_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3

# ── Llama 3 7B LS Rank Ablation ─────────────────────────────────────────────
#
# Tests 4 rank values [128, 64, 32, 16] for the LS model variant at 7B scale
# to find the quality/compression sweet spot before scaling to 405B.
#
# Submit:   sbatch examples/advanced/llm/llama_7b_rank_ablation.sbatch
# Monitor:  squeue -u $USER | grep rank
# Cancel:   scancel <arrayid>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Map array index to rank ─────────────────────────────────────────────────
RANKS=(128 64 32 16)
RANK="${RANKS[$SLURM_ARRAY_TASK_ID]}"

WORKDIR=/scratch/$USER/iterativennsimple
cd "$WORKDIR"
mkdir -p results logs checkpoints/7b_rank_ablation

echo "=============================================="
echo " Job Array  : $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo " Model      : ls (rank=$RANK)"
echo " Node       : $(hostname)"
echo " GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " Start time : $(date)"
echo "=============================================="

# ── Ensure uv is on PATH ─────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# ── Run training ──────────────────────────────────────────────────────────────
uv run --extra llm examples/advanced/llm/llama_comparison.py \
    --config 7b \
    --dataset wikitext-103 \
    --amp --tf32 --gradient-checkpointing \
    --batch-size 1 --grad-accum-steps 32 \
    --lr 3e-4 --weight-decay 0.1 --grad-clip 1.0 \
    --epochs 10 \
    --models ls \
    --rank "$RANK" \
    --checkpoint-dir checkpoints/7b_rank_ablation \
    --log "results/llama_7b_rank_ablation.jsonl" \
    --save-json "results/llama_7b_rank_${RANK}.json" \
    2>&1 | tee "logs/llama_7b_rank${RANK}_${SLURM_ARRAY_JOB_ID}.log"

echo ""
echo "=============================================="
echo " Finished:  ls (rank=$RANK)"
echo " Time:      $(date)"
echo " Result:    results/llama_7b_rank_${RANK}.json"
echo "=============================================="
