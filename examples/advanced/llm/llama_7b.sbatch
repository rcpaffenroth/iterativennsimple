#!/bin/bash
#SBATCH --job-name=llama-7b
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/llama_7b_%A_%a.out
#SBATCH --error=logs/llama_7b_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-4

# ── Llama 3 7B Structured-Sparse Comparison (Job Array) ─────────────────────
#
# Submits 5 jobs (array 0-4), each training one model variant on its own
# RTX PRO 6000 Blackwell (96GB). All 5 run simultaneously.
#
# Memory estimate per variant (bf16 + grad checkpointing, batch=1):
#   Parameters:   ~14 GB (bf16)
#   Optimizer:    ~56 GB (AdamW fp32 master + momentum + variance)
#   Gradients:    ~14 GB (bf16)
#   Activations:  ~5-10 GB (grad checkpointing, batch=1, ctx=4096)
#   Total:        ~89-94 GB → fits on 96GB RTX PRO 6000
#
# Dataset: WikiText-103 (~103M tokens) — meaningful challenge for 7B.
# Checkpoints saved every epoch — resubmit to resume after wall time.
#
# Submit:   sbatch examples/advanced/llm/llama_7b.sbatch
# Monitor:  squeue -u $USER | grep llama
# Logs:     tail -f logs/llama_7b_<arrayid>_<taskid>.out
# Resume:   resubmit — auto-resumes from checkpoints
# Cancel:   scancel <arrayid>   (cancels all 5)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Map array index to model name ─────────────────────────────────────────────
MODELS=(standard ls ls-factored ls-blockdiag ls-blockdiag-factored)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

WORKDIR=/scratch/$USER/iterativennsimple
cd "$WORKDIR"
mkdir -p results logs checkpoints/7b

echo "=============================================="
echo " Job Array  : $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo " Model      : $MODEL"
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
    --models "$MODEL" \
    --checkpoint-dir checkpoints/7b \
    --log "results/llama_7b_103.jsonl" \
    --save-json "results/llama_7b_103_${MODEL}.json" \
    2>&1 | tee "logs/llama_7b_${MODEL}_${SLURM_ARRAY_JOB_ID}.log"

echo ""
echo "=============================================="
echo " Finished:  $MODEL"
echo " Time:      $(date)"
echo " Result:    results/llama_7b_103_${MODEL}.json"
echo " Checkpoint: checkpoints/7b/${MODEL}_final.pt"
echo "=============================================="
