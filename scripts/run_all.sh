#!/bin/bash
# GTS++ Artifact Evaluation: Run all experiments on sample data
# Usage: bash scripts/run_all.sh [gpu_arch]
# Example: bash scripts/run_all.sh 80   # A100
#          bash scripts/run_all.sh 89   # RTX 4080
#          bash scripts/run_all.sh 120a # RTX PRO 6000

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$ROOT_DIR/bin/GTSPP"
DATA="$ROOT_DIR/data/sample"
RESULTS="$ROOT_DIR/results"

mkdir -p "$RESULTS"

# Check binary
if [ ! -f "$BIN" ]; then
    echo "Error: Binary not found at $BIN. Please build first:"
    echo "  cd $ROOT_DIR && mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

# Check sample data
if [ ! -f "$DATA/sift_10k.txt" ]; then
    echo "Generating sample data..."
    python3 "$ROOT_DIR/tools/gen_sample_data.py"
fi

echo "============================================"
echo " GTS++ Artifact Evaluation"
echo "============================================"
echo ""

# ============================================
# Experiment 1: KNN Search (Table 5 / Figure 5)
# ============================================
echo ">>> Experiment 1: KNN Search with Calibrated Pruning"
echo "    (Reproduces Table 5: search speedup across datasets)"
echo ""

for dataset in sift_10k deep_10k gist_1k; do
    echo "--- Dataset: $dataset ---"
    $BIN "$DATA/${dataset}.txt" "$DATA/${dataset}_query.txt" 0 8 \
         "$RESULTS/${dataset}_knn_k8.txt" vec "$DATA/${dataset}_groundtruth.txt"
    echo ""
done

# ============================================
# Experiment 2: KNN with different k values
# ============================================
echo ">>> Experiment 2: Varying k (Table 6)"
echo ""

for k in 1 4 8 16 32; do
    echo "--- SIFT-10K, k=$k ---"
    $BIN "$DATA/sift_10k.txt" "$DATA/sift_10k_query.txt" 0 $k \
         "$RESULTS/sift_10k_knn_k${k}.txt" vec "$DATA/sift_10k_groundtruth.txt"
    echo ""
done

# ============================================
# Experiment 3: Range Query
# ============================================
echo ">>> Experiment 3: Range Query"
echo ""

echo "--- SIFT-10K, r=500 ---"
$BIN "$DATA/sift_10k.txt" "$DATA/sift_10k_qid.txt" 1 500 \
     "$RESULTS/sift_10k_rnn_r500.txt"
echo ""

# ============================================
# Experiment 4: Dynamic Updates (Table 8)
# ============================================
echo ">>> Experiment 4: Dynamic Updates (Insert/Delete)"
echo ""

echo "--- SIFT-10K, mixed workload ---"
$BIN "$DATA/sift_10k.txt" "$DATA/sift_10k_update.txt" 2 0.82 \
     "$RESULTS/sift_10k_update.txt"
echo ""

# ============================================
# Experiment 5: Edit Distance (Word dataset)
# ============================================
echo ">>> Experiment 5: Edit Distance (String Metric)"
echo ""

echo "--- Word-5K ---"
$BIN "$DATA/word_5k.txt" "$DATA/word_5k_qid.txt" 0 8 \
     "$RESULTS/word_5k_knn_k8.txt"
echo ""

# ============================================
# Experiment 6: Cosine Distance
# ============================================
echo ">>> Experiment 6: Cosine Distance"
echo ""

echo "--- Cosine-1K ---"
$BIN "$DATA/cosine_1k.txt" "$DATA/cosine_1k_qid.txt" 0 8 \
     "$RESULTS/cosine_1k_knn_k8.txt"
echo ""

echo "============================================"
echo " All experiments complete!"
echo " Results saved to: $RESULTS/"
echo "============================================"
