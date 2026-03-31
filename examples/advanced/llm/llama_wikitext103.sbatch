#!/bin/bash
#SBATCH --job-name=llama-103
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/llama_103_%j.out
#SBATCH --error=logs/llama_103_%j.err
#SBATCH --mail-type=END,FAIL

# ── Llama 3 WikiText-103 Training (All 5 Variants, Small → Large) ───────────
#
# Runs all 5 structured-sparse variants across configs on 2 GPUs in parallel.
# Each GPU runs one model at a time; we cycle through all variants per config.
#
# With checkpointing, safe to resume after preemption — just resubmit.
#
# Submit:  sbatch examples/advanced/llm/llama_wikitext103.sbatch
# Monitor: tail -f logs/llama_103_<jobid>.out
# Resume:  resubmit (auto-resumes from checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

WORKDIR=/scratch/$USER/iterativennsimple
cd "$WORKDIR"
mkdir -p results logs checkpoints/small checkpoints/medium checkpoints/xlarge checkpoints/large

echo "=============================================="
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $(hostname)"
echo " GPUs       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd', ')"
echo " Start time : $(date)"
echo "=============================================="

export PATH="$HOME/.local/bin:$PATH"

RUN="uv run --extra llm"
SCRIPT="examples/advanced/llm/llama_comparison.py"
DATASET="--dataset wikitext-103"
AMP="--amp --tf32"
MODELS="standard ls ls-factored ls-blockdiag ls-blockdiag-factored"

# Helper: run a model on a specific GPU
run_model() {
    local GPU=$1 CONFIG=$2 MODEL=$3 EPOCHS=$4 BATCH=$5 EXTRA="${6:-}"
    CUDA_VISIBLE_DEVICES=$GPU $RUN $SCRIPT \
        --config "$CONFIG" --epochs "$EPOCHS" $AMP $DATASET \
        --batch-size "$BATCH" \
        --gradient-checkpointing \
        --checkpoint-dir "checkpoints/${CONFIG}" \
        --log "results/llama_${CONFIG}_103.jsonl" \
        --models "$MODEL" \
        --save-json "results/llama_${CONFIG}_103_${MODEL}.json" \
        $EXTRA \
        2>&1 | tee -a "logs/llama_${CONFIG}_${MODEL}_${SLURM_JOB_ID}.log"
}

# ── Small config (~2h per variant) ───────────────────────────────────────────
echo ""
echo ">>> [$(date)] Starting SMALL config (all 5 variants, 2 GPUs)"
for MODEL in $MODELS; do
    # Alternate GPUs: even-index models on GPU 0, odd on GPU 1
    GPU=$(( $(echo $MODELS | tr ' ' '\n' | grep -n "^${MODEL}$" | cut -d: -f1) % 2 ))
    run_model $GPU small "$MODEL" 30 32 &
    # Run at most 2 in parallel
    if [[ $(jobs -r | wc -l) -ge 2 ]]; then
        wait -n
    fi
done
wait
echo ">>> [$(date)] SMALL config done"

# ── Medium config (~6h per variant) ──────────────────────────────────────────
echo ""
echo ">>> [$(date)] Starting MEDIUM config (all 5 variants, 2 GPUs)"
for MODEL in $MODELS; do
    GPU=$(( $(echo $MODELS | tr ' ' '\n' | grep -n "^${MODEL}$" | cut -d: -f1) % 2 ))
    run_model $GPU medium "$MODEL" 20 16 &
    if [[ $(jobs -r | wc -l) -ge 2 ]]; then
        wait -n
    fi
done
wait
echo ">>> [$(date)] MEDIUM config done"

# ── XLarge config (~12h per variant) ─────────────────────────────────────────
echo ""
echo ">>> [$(date)] Starting XLARGE config (all 5 variants, 2 GPUs)"
for MODEL in $MODELS; do
    GPU=$(( $(echo $MODELS | tr ' ' '\n' | grep -n "^${MODEL}$" | cut -d: -f1) % 2 ))
    run_model $GPU xlarge "$MODEL" 15 8 "--grad-accum-steps 2" &
    if [[ $(jobs -r | wc -l) -ge 2 ]]; then
        wait -n
    fi
done
wait
echo ">>> [$(date)] XLARGE config done"

# ── Large config (~24h per variant) ──────────────────────────────────────────
echo ""
echo ">>> [$(date)] Starting LARGE config (all 5 variants, 2 GPUs)"
for MODEL in $MODELS; do
    GPU=$(( $(echo $MODELS | tr ' ' '\n' | grep -n "^${MODEL}$" | cut -d: -f1) % 2 ))
    run_model $GPU large "$MODEL" 10 4 "--grad-accum-steps 8" &
    if [[ $(jobs -r | wc -l) -ge 2 ]]; then
        wait -n
    fi
done
wait
echo ">>> [$(date)] LARGE config done"

# ── Generate plots ────────────────────────────────────────────────────────────
echo ""
echo ">>> [$(date)] Generating comparison plots"
for CONFIG in small medium xlarge large; do
    if [[ -f "results/llama_${CONFIG}_103.jsonl" ]]; then
        $RUN $SCRIPT --plot-from "results/llama_${CONFIG}_103.jsonl" \
            --plot-dir "results/plots_${CONFIG}"
    fi
done

echo ""
echo "=============================================="
echo " All training complete: $(date)"
echo " Results:     results/llama_*_103_*.json"
echo " Logs:        results/llama_*_103.jsonl"
echo " Plots:       results/plots_*/"
echo " Checkpoints: checkpoints/*/"
echo "=============================================="
