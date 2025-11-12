# HPFL: Hierarchical Cluster-based Personalized Federated Learning

A comprehensive implementation of Hierarchical Cluster-based Personalized Federated Learning (HPFL) for Satellite-Aerial-Ground Integrated Networks (SAGIN). This framework addresses data and system heterogeneity in federated learning through dynamic client selection (DCS) and model similarity-based clustering.

## Overview

This project implements a hierarchical federated learning system that combines:
- **Dynamic Client Selection (DCS)**: Selects clients based on communication quality, computational capability, data significance, and contribution
- **Model Similarity-based Clustering**: Groups clients with similar model updates to handle data heterogeneity
- **Personalized Learning**: Provides cluster-specific models for improved accuracy on non-IID data

## Project Structure

```
hpfl/
├── src/                    # Source code
│   ├── core/              # Core modules
│   │   ├── data.py        # FEMNIST dataset loading and writer-based partitioning
│   │   ├── model.py       # Neural network model (SimpleCNN)
│   │   ├── network.py     # Network simulation (Link, CostMeter)
│   │   ├── fl_nodes.py    # FL nodes (FLClient, UAV, SatAgg)
│   │   ├── aggregation.py # Aggregation functions (fedavg)
│   │   └── clustering.py  # Model similarity-based clustering
│   ├── utils/             # Utilities
│   │   ├── model_utils.py # Model-related utilities (state_dict conversion)
│   │   └── progress_logger.py # Progress logging system
│   ├── config/            # Configuration
│   │   └── default_config.py  # Default experiment settings
│   └── experiments/       # Experiment scripts
│       ├── bl1.py         # BL1: Hierarchical FedAvg (baseline)
│       ├── bl2.py         # BL2: DCS Only
│       ├── bl3.py         # BL3: Clustering Only
│       ├── abl1.py        # ABL-1: DCS + Clustering (proposed method)
│       └── abl2.py        # ABL-2: DCS + Clustering + Head-Personalization
├── scripts/               # Execution scripts
│   ├── run_experiment.sh  # Run individual experiment
│   ├── run_all.sh         # Run all experiments sequentially
│   ├── clear.sh           # Stop running experiments
│   └── install_dependencies.sh # Install dependencies
├── logs/                  # Experiment logs
│   ├── bl1/              # BL1 experiment logs
│   ├── bl2/              # BL2 experiment logs
│   ├── bl3/              # BL3 experiment logs
│   ├── abl1/             # ABL-1 experiment logs
│   ├── progress/         # Progress logs
│   └── timestamp/        # Timestamp logs
├── results/              # Experiment results
├── materials/            # Research materials
└── requirements.txt      # Python dependencies
```

## Core Modules

### `src/core/data.py`
- **FEMNISTDataset**: Wraps Hugging Face FEMNIST dataset as PyTorch Dataset
- **setup_femnist_by_writer()**: Partitions data by writer ID to create non-IID distribution
  - Each client receives data from multiple writers
  - Automatic train/test split
  - Progress logging with tqdm

### `src/core/model.py`
- **SimpleCNN**: Lightweight CNN for FEMNIST classification
  - Architecture: Conv2d(1→32) → MaxPool → Conv2d(32→64) → MaxPool → FC(128) → FC(62)
  - 62 classes (uppercase, lowercase letters, digits)

### `src/core/network.py`
- **Link**: Network link simulation
  - Models latency, jitter, bandwidth, and packet loss
  - `transmit()`: Simulates packet transmission with success/failure and delay
- **CostMeter**: Communication and time cost measurement
  - Tracks round-by-round statistics
  - Calculates cumulative costs (total bytes, total time)
  - `end_round()`: Finalizes round costs

### `src/core/fl_nodes.py`
- **Packet**: Network packet structure (src, dst, timestamp, data, type)
- **FLClient**: Federated learning client
  - DCS state: communication quality, computational capability, data significance, contribution
  - `calculate_dcs_score()`: Computes DCS score (α·qi + β·ci + γ·di + δ·gi)
  - `train()`: Performs local training (uses global or cluster model)
- **UAV**: Regional aggregator
  - `assigned_clients`: List of assigned clients
  - `select_clients_dcs()`: Selects clients using DCS
  - `run()`: Processes packets in batches (8 packets per batch)
  - `flush()`: Sends batched packets to satellite
- **SatAgg**: Satellite aggregator
  - `buffer`: Stores collected model updates
  - `run()`: Collects model updates loop

### `src/core/aggregation.py`
- **fedavg()**: Federated Averaging
  - Averages multiple model state_dicts
  - Supports weighted averaging

### `src/core/clustering.py`
- **extract_model_features()**: Extracts feature vectors from model state_dicts
- **compute_similarity_matrix()**: Computes cosine similarity matrix between models
- **cluster_models()**: Performs model similarity-based clustering using Agglomerative Clustering
- **cluster_based_aggregation()**: Aggregates models within each cluster

### `src/utils/progress_logger.py`
- **ProgressLogger**: Progress logging system
  - Logs to file and console simultaneously
  - Timestamp tracking
  - Real-time progress display

## Experiment Models

### BL1: Hierarchical FedAvg (Baseline)
- **Description**: Basic hierarchical federated learning
- **Features**: 
  - Random client selection
  - Simple FedAvg aggregation
  - No DCS or clustering
- **Purpose**: Baseline performance and cost measurement
- **Target**: Global Accuracy ≥ 87%

### BL2: DCS Only
- **Description**: Dynamic Client Selection only
- **Features**:
  - DCS-based client selection at UAV level
  - No clustering applied
  - Handles system heterogeneity
- **Purpose**: Measure system heterogeneity management effect (cost reduction)
- **Target**: Global Accuracy ≥ 87%

### BL3: Clustering Only
- **Description**: Model similarity-based clustering only
- **Features**:
  - Random client selection
  - Model similarity-based clustering
  - Cluster-specific model aggregation
  - No DCS applied
- **Purpose**: Measure data heterogeneity mitigation effect (accuracy improvement)
- **Target**: Personalized Accuracy ≥ 100%

### ABL-1: HPFL (Proposed Method)
- **Description**: DCS + Clustering
- **Features**:
  - DCS-based client selection at UAV level
  - Model similarity-based clustering
  - Cluster-specific model aggregation
  - Combines both strategies
- **Purpose**: Verify synergy effect of both strategies (cost reduction + accuracy improvement)
- **Target**: Personalized Accuracy ≥ 100%

### ABL-2: HPFL with Head-Personalization
- **Description**: DCS + Clustering + Head-Personalization
- **Features**:
  - DCS-based client selection
  - Model similarity-based clustering
  - Head-only training (backbone frozen, only classifier trained)
  - Enhanced personalization
- **Purpose**: Further improve personalization with head-only training
- **Target**: Personalized Accuracy ≥ 100%

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Install Dependencies

```bash
# Using the installation script
./scripts/install_dependencies.sh

# Or manually
pip install -r requirements.txt
```

### Dependencies
- `torch>=1.9.0`: PyTorch for deep learning
- `torchvision>=0.10.0`: Vision utilities
- `datasets>=2.0.0`: Hugging Face datasets (for FEMNIST)
- `scikit-learn>=1.0.0`: Clustering algorithms
- `numpy>=1.21.0`: Numerical operations
- `tqdm>=4.60.0`: Progress bars

## Running Experiments

### Individual Experiment

```bash
# Run BL1 experiment
./scripts/run_experiment.sh bl1

# Run BL2 experiment
./scripts/run_experiment.sh bl2

# Run BL3 experiment
./scripts/run_experiment.sh bl3

# Run ABL-1 experiment
./scripts/run_experiment.sh abl1

# Run ABL-2 experiment
./scripts/run_experiment.sh abl2
```

Or directly with Python:

```bash
# BL1
python -m src.experiments.bl1

# BL2
python -m src.experiments.bl2

# BL3
python -m src.experiments.bl3

# ABL-1
python -m src.experiments.abl1

# ABL-2
python -m src.experiments.abl2
```

### Run All Experiments Sequentially

```bash
./scripts/run_all.sh
```

This will run BL1 → BL2 → BL3 → ABL-1 → ABL-2 sequentially, waiting for each to complete before starting the next.

### Stop Experiments

```bash
# Stop specific experiment
./scripts/clear.sh bl1
./scripts/clear.sh bl2
./scripts/clear.sh bl3
./scripts/clear.sh abl1
./scripts/clear.sh abl2

# Stop all experiments
./scripts/clear.sh all
```

## Monitoring Experiments

### View Logs

```bash
# Real-time log monitoring
tail -f logs/bl1/bl1_YYYYMMDD_HHMMSS.log
tail -f logs/bl2/bl2_YYYYMMDD_HHMMSS.log
tail -f logs/bl3/bl3_YYYYMMDD_HHMMSS.log
tail -f logs/abl1/abl1_YYYYMMDD_HHMMSS.log
tail -f logs/abl2/abl2_YYYYMMDD_HHMMSS.log

# List recent logs
ls -lt logs/bl1/ | head -5
ls -lt logs/bl2/ | head -5
ls -lt logs/bl3/ | head -5
ls -lt logs/abl1/ | head -5
ls -lt logs/abl2/ | head -5

# View progress logs
tail -f logs/progress/latest.log
```

### Log Structure

Each experiment generates:
- **Main log**: `logs/{experiment}/{experiment}_{timestamp}.log` - Complete experiment output
- **Progress log**: `logs/progress/latest.log` - Progress tracking
- **Timestamp log**: `logs/timestamp/latest.log` - Timestamp tracking
- **PID file**: `logs/{experiment}.pid` - Process ID for management

## Evaluation Metrics

### Performance Metrics

- **Global Accuracy (GA)**: Accuracy on the global test set (concatenated client test sets)
- **Personalized Accuracy (PA)**: Average accuracy on each client's local test set
- **Global Loss**: Cross-entropy loss on global test set
- **Personalized Loss**: Average cross-entropy loss on client local test sets

### Efficiency Metrics

- **Round Communication Cost**: Data transmitted in each round (KB)
- **Round Time Cost**: Simulated time for each round (seconds)
- **Total Communication Cost**: Cumulative data transmitted until target accuracy (MB/GB)
- **Total Time Cost**: Cumulative simulated time until target accuracy (seconds)

### Output Format

Each round outputs:
```
=== Round N ===
  Received X updates.
  Computing model similarity and clustering...
  Clustered into Y clusters: {...}
  Updated Y cluster models.
  [Perf] Global GA: XX.XX% | Global Loss: X.XXXX
  [Perf] Personalized PA: XX.XX% | Personalized Loss: X.XXXX (X clients)
  [Effi] Round Cost: XXX KB, +X.XXs simulated time
  [Cumul] Total Data: XX.XX MB | Total Time: XX.XXs
```

## Configuration

All experiment settings are in `src/config/default_config.py`:

### Network Topology
- `NUM_UAV`: Number of UAVs (default: 6)
- `NUM_CLIENTS`: Total number of clients (default: 100)

### Training Settings
- `ROUNDS`: Maximum number of rounds (default: 50)
- `LOCAL_EPOCHS`: Local training epochs per round (default: 1)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LR`: Learning rate (default: 0.01)
- `MOMENTUM`: Momentum for SGD (default: 0.9)

### Client Selection
- `SAMPLE_FRAC`: Fraction of clients participating per round (default: 0.1, i.e., 10%)

### Target Accuracy
- `TARGET_ACC`: Target Global Accuracy for baseline experiments (default: 87.0%)
- `CLUSTERING_TARGET_ACC`: Target Personalized Accuracy for clustering experiments (default: 100.0%)

### Network Bandwidth
- `CLIENT_UPLINK_BW`: Client-to-UAV uplink bandwidth in bps (default: 200,000)
- `UAV_SAT_BW`: UAV-to-Satellite bandwidth in bps (default: 600,000)

### DCS Weights
- `ALPHA`: Communication quality weight (default: 0.25)
- `BETA`: Computational capability weight (default: 0.25)
- `GAMMA`: Data significance weight (default: 0.25)
- `DELTA`: Contribution weight (default: 0.25)

### Device
- `DEVICE`: Computing device ("cuda" or "cpu", auto-detected)
- `SEED`: Random seed for reproducibility (default: 42)

## Experiment Workflow

### Typical Experiment Flow

1. **Initialization**
   - Load FEMNIST dataset from Hugging Face
   - Partition data by writer ID (non-IID distribution)
   - Initialize network topology (clients, UAVs, satellite)
   - Initialize global/cluster models

2. **Each Round**
   - **Client Selection**: Random (BL1, BL3) or DCS-based (BL2, ABL-1, ABL-2)
   - **Local Training**: Selected clients train on local data
   - **Model Transmission**: Clients send updates to assigned UAVs
   - **UAV Aggregation**: UAVs batch and forward to satellite
   - **Clustering** (BL3, ABL-1, ABL-2): Compute similarity and cluster models
   - **Aggregation**: Aggregate models (global or cluster-specific)
   - **Evaluation**: Compute Global and Personalized metrics
   - **Cost Measurement**: Track communication and time costs

3. **Termination**
   - Stop when target accuracy is reached or max rounds exceeded
   - Output final metrics and costs

## Key Features

### Non-IID Data Distribution
- Writer-based partitioning ensures realistic non-IID distribution
- Each client has data from multiple writers with similar writing styles

### Hierarchical Architecture
- Three-tier hierarchy: Clients → UAVs → Satellite
- Efficient batch processing at UAV level
- Realistic network simulation with latency, jitter, and packet loss

### Dynamic Client Selection
- Multi-factor scoring: communication quality, computational capability, data significance, contribution
- Adapts to system heterogeneity
- Reduces communication costs

### Model Clustering
- Cosine similarity-based clustering
- Cluster-specific model aggregation
- Handles data heterogeneity effectively

### Progress Tracking
- Real-time progress logging
- Timestamp tracking
- Comprehensive experiment logs

## Results

Experiment results are saved in the `results/` directory. Each experiment tracks:
- Convergence curves (accuracy vs. rounds)
- Communication cost vs. accuracy
- Time cost vs. accuracy
- Comparison between different methods

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
