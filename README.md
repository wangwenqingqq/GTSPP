# GTS++: Efficient Dynamic Similarity Search in General Metric Spaces on GPUs

Artifact for SIGMOD 2027 submission. GTS++ is a GPU-accelerated index for similarity search over **arbitrary metric spaces** (L2, cosine, edit distance, and any user-defined metric satisfying the triangle inequality) with real-time dynamic updates.

## Key Features

- **Per-level calibrated pruning (C1)**: Tightens the triangle-inequality lower bound by a per-level scale factor γ\_l ≥ 1, calibrated via binary search to preserve ≥99.9% recall. Cost: 2 FLOPs/node.
- **Cost-model-driven buffer (C2)**: Replaces frequent full rebuilds with O(log n) incremental inserts via leaf-routed overflow pools.
- **GPU workspace reuse (C3)**: Pre-allocated buffers eliminate per-query cudaMalloc/cudaFree overhead.

## Requirements

- **CUDA Toolkit** ≥ 12.0
- **CMake** ≥ 3.18
- **GPU**: Any NVIDIA GPU with compute capability ≥ 7.0 (tested on A100 sm_80, RTX 4080 sm_89, RTX PRO 6000 sm_120a)
- **Python 3** (for sample data generation only)

## Quick Start

### 1. Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80    # Set your GPU arch: 80(A100), 89(4080), 120a(PRO6000)
make -j$(nproc)
cd ..
```

The binary is produced at `bin/GTSPP`.

### 2. Generate Sample Data

```bash
python3 tools/gen_sample_data.py
```

This creates small-scale datasets under `data/sample/`:
| Dataset | Dim | Size | Metric | Distance Code |
|---------|-----|------|--------|---------------|
| sift\_10k | 128 | 10K | L2 | 2 |
| gist\_1k | 960 | 1K | L2 | 2 |
| deep\_10k | 96 | 10K | L2 | 2 |
| cosine\_1k | 300 | 1K | Cosine | 5 |
| word\_5k | var | 5K | Edit dist | 6 |

### 3. Run Experiments

**All experiments at once:**
```bash
bash scripts/run_all.sh
```

**Individual runs:**
```bash
# KNN search (process_type=0), k=8, with recall evaluation
bin/GTSPP data/sample/sift_10k.txt data/sample/sift_10k_query.txt 0 8 results/output.txt vec data/sample/sift_10k_groundtruth.txt

# Range query (process_type=1), radius=500
bin/GTSPP data/sample/sift_10k.txt data/sample/sift_10k_qid.txt 1 500 results/output.txt

# Dynamic update (process_type=2), radius=0.82
bin/GTSPP data/sample/sift_10k.txt data/sample/sift_10k_update.txt 2 0.82 results/output.txt
```

### 4. Ablation Study (Table 12)

```bash
bash scripts/run_ablation.sh
```

Compares baseline (no calibration) vs. per-level calibrated pruning (C1). Observe the search time difference and recall in the output.

## Command Reference

```
bin/GTSPP <data_file> <query_file> <process_type> <k_or_r> <output_file> [query_mode] [groundtruth_file]
```

| Argument | Description |
|----------|-------------|
| `data_file` | Dataset in GTS text format (see below) |
| `query_file` | Query IDs (text) or query vectors (text/binary) |
| `process_type` | `0` = KNN, `1` = range query, `2` = dynamic update |
| `k_or_r` | k for KNN, radius r for range/update |
| `output_file` | Path to write timing and result statistics |
| `query_mode` | `id` (default) or `vec` for vector queries |
| `groundtruth_file` | Optional. Enables recall evaluation and per-level calibration |

## Data Format

**Vector dataset** (GTS text format):
```
<dim> <num_vectors> <distance_code>
<v_1_1> <v_1_2> ... <v_1_dim>
<v_2_1> <v_2_2> ... <v_2_dim>
...
```

Distance codes: `0`=L∞, `1`=L1, `2`=L2, `5`=Cosine, `6`=Edit distance.

**String dataset** (edit distance):
```
<max_string_length> <num_strings> 6
string1
string2
...
```

**Query ID file**:
```
<num_queries>
<id_1>
<id_2>
...
```

**Ground truth file** (text): one line per query, space-separated IDs of top-100 nearest neighbors.

**Update workload file**:
```
<num_operations> <radius>
<flag> <object_id>     # flag: 1=insert, 0=delete
...
```

## Reproducing Paper Results

### Full-scale datasets

For full-scale evaluation (Tables 5–11, Figures 5–8), download:

- **SIFT-1M, GIST-1M, Deep-1M**: [ANN-Benchmarks](http://ann-benchmarks.com/)
- **Word (611K strings)**: Available from the GTS repository
- **T-loc 1M/10M, Vector 200K**: Available from the GTS repository

Convert `.fvecs`/`.ivecs` to GTS text format using `tools/gen_sample_data.py` as reference, or load binary files directly (the system supports both).

### Key parameters (as in paper)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Tree order (C) | 10 | Branching factor |
| Max leaf size | 20 | Maximum objects per leaf node |
| Calibration queries | 1000 | Subset size for per-level calibration |
| Safety margin | 0.70 | Scale reduction: γ\_safe = 1 + (γ\_raw − 1) × 0.70 |
| Target recall | 99.9% | Calibration target |
| Buffer size (B\*) | 300 | Optimal from cost model (§4.3) |
| Overflow pool | 64 | Per-leaf overflow capacity |

## Project Structure

```
GTSPP/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── src/
│   └── main.cu             # Entry point: index build, search, update, calibration
├── include/
│   ├── config.cuh           # Global constants and compile-time configuration
│   ├── file.cuh             # Data I/O (text and binary formats)
│   ├── gpu_timer.cuh        # CUDA event-based timing utilities
│   ├── tree.cuh             # GPU-parallel tree construction
│   ├── search_v2.cuh        # KNN and range search engine (~2200 lines)
│   ├── search.cuh           # Legacy search (used by update path)
│   ├── search_naive.cuh     # Brute-force baseline
│   ├── update.cuh           # Dynamic insert/delete operations
│   ├── residual_pruning.cuh # Per-level scale-based pruning (C1, device functions)
│   ├── residual_tuner.cuh   # Calibration and online adaptation (C1, host class)
│   └── mlp_constant.cuh     # Legacy MLP declarations (superseded by scale pruning)
├── tools/
│   └── gen_sample_data.py   # Sample dataset generator
├── scripts/
│   ├── run_all.sh           # Run all experiments
│   └── run_ablation.sh      # Pruning ablation study
└── data/
    └── sample/              # Generated sample datasets
```

## Supported Distance Metrics

| Code | Metric | Input Type |
|------|--------|------------|
| 0 | L∞ (Chebyshev) | Fixed-dim vectors |
| 1 | L1 (Manhattan) | Fixed-dim vectors |
| 2 | L2 (Euclidean) | Fixed-dim vectors |
| 5 | Cosine angle | Fixed-dim vectors |
| 6 | Edit distance | Variable-length strings |

User-defined metrics can be added by extending the distance computation in `tree.cuh`, `search_v2.cuh`, and `update.cuh`.

## License

This artifact is provided for academic review purposes.
