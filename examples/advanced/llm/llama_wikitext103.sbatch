#!/bin/bash
#SBATCH --job-name=llama-103
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=128000
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=logs/llama_103_%j.out
#SBATCH --error=logs/llama_103_%j.err
#SBATCH --mail-type=END,FAIL

# ── Llama 3 WikiText-103 Training (Weekend Run) ─────────────────────────────
# Runs standard vs LSLinear Llama on both GPUs in parallel across all configs.
#
# Submit:  sbatch scripts/slurm/llama_wikitext103.sbatch
# Monitor: tail -f logs/llama_103_<jobid>.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd /scratch/jrmolnia/iterativennsimple
mkdir -p results logs

echo "=============================================="
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $(hostname)"
echo " GPUs       : $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ')"
echo " Start time : $(date)"
echo "=============================================="

# ── Ensure uv is on PATH ────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# Common flags
RUN="uv run --extra llm"
DATASET="--dataset wikitext-103"
AMP="--amp"

# ── Small config: both models on separate GPUs (~2 hours) ───────────────────
echo ""
echo ">>> [$(date)] Starting SMALL config (standard on GPU0, ls on GPU1)"

CUDA_VISIBLE_DEVICES=0 $RUN examples/llm/llama_comparison.py \
    --config small --epochs 30 $AMP $DATASET \
    --models standard \
    --save-json results/llama_small_103_standard.json \
    2>&1 | tee logs/llama_small_103_standard.log &
PID_SMALL_STD=$!

CUDA_VISIBLE_DEVICES=1 $RUN examples/llm/llama_comparison.py \
    --config small --epochs 30 $AMP $DATASET \
    --models ls \
    --save-json results/llama_small_103_ls.json \
    2>&1 | tee logs/llama_small_103_ls.log &
PID_SMALL_LS=$!

wait $PID_SMALL_STD $PID_SMALL_LS
echo ">>> [$(date)] SMALL config done"

# ── Medium config: both models on separate GPUs (~8 hours) ──────────────────
echo ""
echo ">>> [$(date)] Starting MEDIUM config (standard on GPU0, ls on GPU1)"

CUDA_VISIBLE_DEVICES=0 $RUN examples/llm/llama_comparison.py \
    --config medium --epochs 20 $AMP $DATASET \
    --models standard \
    --save-json results/llama_medium_103_standard.json \
    2>&1 | tee logs/llama_medium_103_standard.log &
PID_MED_STD=$!

CUDA_VISIBLE_DEVICES=1 $RUN examples/llm/llama_comparison.py \
    --config medium --epochs 20 $AMP $DATASET \
    --models ls \
    --save-json results/llama_medium_103_ls.json \
    2>&1 | tee logs/llama_medium_103_ls.log &
PID_MED_LS=$!

wait $PID_MED_STD $PID_MED_LS
echo ">>> [$(date)] MEDIUM config done"

# ── Large config: both models on separate GPUs (~24 hours) ──────────────────
echo ""
echo ">>> [$(date)] Starting LARGE config (standard on GPU0, ls on GPU1)"

CUDA_VISIBLE_DEVICES=0 $RUN examples/llm/llama_comparison.py \
    --config large --epochs 10 $AMP $DATASET \
    --batch-size 16 \
    --models standard \
    --save-json results/llama_large_103_standard.json \
    2>&1 | tee logs/llama_large_103_standard.log &
PID_LRG_STD=$!

CUDA_VISIBLE_DEVICES=1 $RUN examples/llm/llama_comparison.py \
    --config large --epochs 10 $AMP $DATASET \
    --batch-size 16 \
    --models ls \
    --save-json results/llama_large_103_ls.json \
    2>&1 | tee logs/llama_large_103_ls.log &
PID_LRG_LS=$!

wait $PID_LRG_STD $PID_LRG_LS
echo ">>> [$(date)] LARGE config done"

echo ""
echo "=============================================="
echo " All training complete: $(date)"
echo " Results in: results/llama_*_103_*.json"
echo " Logs in:    logs/llama_*_103_*.log"
echo "=============================================="
