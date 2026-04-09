#!/bin/bash
# GTS++ Ablation Study (Table 12 in paper)
# Compares: (1) no pruning, (2) scale-based pruning, (3) full per-level calibration
#
# To toggle pruning mode, we use the ground-truth file presence:
# - With GT file: full calibration (C1 active)
# - Without GT file: baseline (no calibration, scales = 1.0)
#
# For a proper ablation on full-scale data, see Section 5 of the paper.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$ROOT_DIR/bin/GTSPP"
DATA="$ROOT_DIR/data/sample"
RESULTS="$ROOT_DIR/results"

mkdir -p "$RESULTS"

if [ ! -f "$BIN" ]; then
    echo "Error: Binary not found. Build first."
    exit 1
fi

if [ ! -f "$DATA/sift_10k.txt" ]; then
    python3 "$ROOT_DIR/tools/gen_sample_data.py"
fi

echo "============================================"
echo " Ablation Study: Pruning Effectiveness"
echo "============================================"
echo ""

echo ">>> Baseline (no pruning calibration):"
$BIN "$DATA/sift_10k.txt" "$DATA/sift_10k_query.txt" 0 8 \
     "$RESULTS/ablation_baseline.txt" vec
echo ""

echo ">>> With Per-Level Calibration (C1):"
$BIN "$DATA/sift_10k.txt" "$DATA/sift_10k_query.txt" 0 8 \
     "$RESULTS/ablation_calibrated.txt" vec "$DATA/sift_10k_groundtruth.txt"
echo ""

echo "Compare search times above to see C1 speedup."
echo "============================================"
