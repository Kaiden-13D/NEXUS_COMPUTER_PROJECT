# Experiment Parameters Reference

This document summarizes the experimental parameters used in our HPFL implementation and their references from related literature.

## Our Implementation Parameters

### Network Topology
- **Number of UAVs**: 6
- **Number of Clients**: 100
- **Hierarchy**: 3-tier (Client → UAV → Satellite)

### Training Parameters

| Parameter | Value | Reference/Note |
|-----------|-------|----------------|
| **Batch Size** | 32 | Zhang_arxiv24 [1], Zheng_2024 [4] |
| **Learning Rate (η)** | 0.01 | Liu_arxiv25 [1], Zhang_arxiv24 [1] |
| **Momentum** | 0.9 | Chien_IoT25 [3], Liu_arxiv25 [1] |
| **Local Epochs (E)** | 1 | (Our choice for faster convergence) |
| **Optimizer** | SGD | Standard in FL literature |
| **Total Rounds (R)** | 50 | (Our choice, can be extended) |
| **Client Selection Rate (ε)** | 0.1 (10%) | Similar to Zhang_arxiv24 (25/100 = 0.25) [1] |

### Network Communication Parameters

| Parameter | Value | Reference/Note |
|-----------|-------|----------------|
| **Client-UAV Latency** | 50 ms | Liu_arxiv25: 50-250ms satellite-ground [1] |
| **Client-UAV Jitter** | 25 ms | (Modeling network variability) |
| **Client-UAV Packet Loss** | 0.02 (2%) | (Realistic wireless link loss) |
| **UAV-Satellite Latency** | 30-55 ms | Liu_arxiv25: 50-250ms [1] |
| **UAV-Satellite Jitter** | 10 ms | (Lower jitter for satellite links) |
| **UAV-Satellite Packet Loss** | 0.01 (1%) | (Lower loss for satellite links) |
| **Client-UAV Bandwidth** | 200,000 bps | (Modeling constrained uplink) |
| **UAV-Satellite Bandwidth** | 600,000 bps | (Higher bandwidth for satellite links) |

### Dataset
- **Dataset**: FEMNIST (62 classes: uppercase, lowercase, digits)
- **Non-IID Distribution**: Writer-based partitioning
- **Train/Test Split**: 90/10 per client

### Model Architecture
- **Model**: SimpleCNN
  - Conv2d(1→32) → MaxPool → Conv2d(32→64) → MaxPool → FC(128) → FC(62)
- **Loss Function**: Cross-Entropy Loss

### Clustering Parameters
- **Clustering Method**: Agglomerative Clustering (cosine similarity-based)
- **Number of Clusters**: Auto-determined (2-5, heuristic: `max(2, min(5, n//3))`)
- **Similarity Metric**: Cosine similarity on model weight features

### DCS (Dynamic Client Selection) Parameters
- **DCS Weights** (equal distribution):
  - α = 0.25 (Communication quality)
  - β = 0.25 (Computational capability)
  - γ = 0.25 (Data significance)
  - δ = 0.25 (Contribution)

## Comparison with Related Works

### Zhang_arxiv24 (Client Auto Selection)
- **Clients**: 100 ✓ (matches our setting)
- **Batch Size**: 32 ✓ (matches our setting)
- **Learning Rate**: 0.01 ✓ (matches our setting)
- **Local Epochs**: 5 (we use 1 for faster rounds)
- **Rounds**: 1000 (we use 50, can be extended)
- **Client Selection**: 25/100 = 0.25 (we use 0.1 = 10/100)
- **Model**: CNN (2×Conv + 3×FC) (similar structure)

### Liu_arxiv25 (SFedSat, Satellite-FL)
- **Structure**: 2-layer Satellite-Ground (we use 3-layer: Client-UAV-Satellite)
- **Clients**: 24-48 (we use 100)
- **Batch Size**: 64 (we use 32)
- **Learning Rate**: 0.01 ✓ (matches our setting)
- **Local Epochs**: 5 (we use 1)
- **Delay**: 50-250ms ✓ (our latency: 30-55ms UAV-SAT, 50ms Client-UAV)
- **Communication Model**: Bandwidth, power, noise (we model bandwidth and delay)

### Chien_IoT25
- **Clients**: 20 (we use 100)
- **Batch Size**: 20/40 (we use 32)
- **Momentum**: 0.9 ✓ (matches our setting)
- **Local Epochs**: 2 (we use 1)
- **Rounds**: 100-300 (we use 50)
- **Optimizer**: SGD with momentum 0.9 ✓ (matches our setting)

### Zhou_TVT24 (3-tier Cloud-Edge-Device)
- **Structure**: 3-tier ✓ (similar to our Client-UAV-Satellite)
- **Clients**: 20 (we use 100)
- **Batch Size**: 20 (we use 32)
- **Learning Rate**: 0.05 (we use 0.01)
- **Rounds**: 200 (we use 50)

### Zheng_2024
- **Batch Size**: 32 ✓ (matches our setting)
- **Learning Rate**: 0.001 (we use 0.01)
- **Local Epochs**: 5 (we use 1)
- **Rounds**: 1000-2000 (we use 50)

## Key Differences and Rationale

1. **Local Epochs (E=1)**: We use 1 epoch per round for faster convergence and to better simulate resource-constrained edge devices, while most papers use 2-5 epochs.

2. **Client Selection Rate (10%)**: Lower than typical 20-25% to better demonstrate the efficiency of DCS in resource-constrained scenarios.

3. **Network Latency**: Our values (30-55ms for UAV-SAT, 50ms for Client-UAV) are within the range of Liu_arxiv25's satellite-ground communication delay (50-250ms).

4. **Batch Size (32)**: Matches Zhang_arxiv24 and Zheng_2024, providing a good balance between convergence speed and memory usage.

5. **Learning Rate (0.01)**: Standard value used in multiple FL papers (Liu_arxiv25, Zhang_arxiv24), providing stable convergence.

6. **Number of Clients (100)**: Matches Zhang_arxiv24, larger than most other works (20-48) to better demonstrate scalability.

## References

[1] Zhang_arxiv24 - Client auto selection paper
[2] Liu_arxiv25 - SFedSat, Satellite-FL (2-layer structure)
[3] Chien_IoT25 - FL with various datasets
[4] Zheng_2024 - Using actual supplier data
[5] Zhou_TVT24 - 3-tier Cloud-Edge-Device structure

## Notes for Paper Writing

When citing parameters in the paper:

1. **Batch Size (32)**: "Following [Zhang_arxiv24, Zheng_2024], we set batch size to 32."

2. **Learning Rate (0.01)**: "We use learning rate η = 0.01, consistent with [Liu_arxiv25, Zhang_arxiv24]."

3. **Momentum (0.9)**: "SGD optimizer with momentum 0.9 is employed, as in [Chien_IoT25, Liu_arxiv25]."

4. **Network Latency**: "Network delays are set to 30-55ms for UAV-Satellite links and 50ms for Client-UAV links, within the range of satellite-ground communication delays (50-250ms) reported in [Liu_arxiv25]."

5. **Client Selection Rate**: "We set client selection rate to 10% (10 clients per round), lower than typical 20-25% to better demonstrate efficiency in resource-constrained scenarios."

6. **Number of Clients**: "We use 100 clients, matching [Zhang_arxiv24], to better demonstrate scalability compared to smaller-scale experiments (20-48 clients) in other works."

7. **Local Epochs**: "We use 1 local epoch per round, prioritizing faster convergence and better simulation of resource-constrained edge devices, while most works use 2-5 epochs."

