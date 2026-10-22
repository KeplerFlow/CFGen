# Temporal Graph Generation

This directory contains the core implementation for temporal graph generation and future community search.

## Requirements

Python 3.10 or newer is recommended. Install the runtime dependencies with:

```bash
pip install numpy scipy scikit-learn pandas networkx tqdm joblib faiss-cpu
```

Install a compatible PyTorch and PyTorch Geometric build for your CPU or CUDA environment:

```bash
pip install torch
pip install torch-geometric
```

For large graphs, CUDA-enabled PyTorch and GPU Faiss are recommended.

## Data Layout

Create a `datasets/` directory next to `src/`:

```text
cfgen_clean/
├── datasets/
│   ├── example/
│   │   ├── 0.txt
│   │   ├── 1.txt
│   │   └── ...
│   └── example_gt.txt
├── saved_models/
└── src/
```

Each snapshot file is a whitespace-separated edge list. The filename stem must be an integer timestamp. Each non-empty line contains two node IDs:

```text
12 45
45 81
```

The optional `<dataset>_gt.txt` file contains query communities used for evaluation.

## Running the Pipeline

Run all commands from the `cfgen_clean/` directory.

### 1. Coarsening and embedding generation

```bash
python src/main.py --dataset example --ratio 0.95
```

This creates the phase-one data and coarse/golden embedding files in `saved_models/`.

### 2. Train the diffusion model

```bash
python src/train.py \
  --dataset example \
  --batch_size 2048 \
  --horizons 1
```

Use a comma-separated list for multiple forecast horizons, for example `--horizons 1,2,3`.

### 3. Train the refinement model

```bash
python src/train_refinement.py \
  --dataset example \
  --batch_size 2048
```

### 4. Run future community search

```bash
python src/query.py \
  --dataset example \
  --diff_steps 20 \
  --num_queries 1000 \
  --query_size 1 \
  --alpha 0.7
```

Results are written below `logs/`. The query stage expects the phase-one files and both trained model files in `saved_models/`.

## Optional Edge Decoder

The standalone edge decoder can be trained after phase-one embedding generation:

```bash
python src/train_edge_decoder.py \
  --dataset example \
  --device cuda \
  --epochs 12 \
  --horizons 1
```

The resulting checkpoint is saved in `saved_models/`.

## Useful Environment Variables

The query stage supports these optional overrides:

```bash
GRAPH_DIR=/path/to/snapshots
TARGET_COUNT=100
GRAPH_SCOPE=local
LOCAL_GRAPH_THRESHOLD=5000000
```

If `GRAPH_DIR` is not set, snapshots are read from `datasets/<dataset>/`.
