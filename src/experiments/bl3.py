"""
BL3: Clustering Only
- Random client selection
- Model similarity-based clustering applied
- Cluster-based model aggregation
- No DCS applied
- Measure data heterogeneity mitigation effect (accuracy improvement)
"""
import asyncio
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ..core import (
    setup_femnist_by_writer, SimpleCNN, Link, CostMeter,
    FLClient, UAV, SatAgg, Packet, fedavg,
    compute_similarity_matrix, cluster_models, cluster_based_aggregation
)
from ..utils import get_state_dict_bytes
from ..utils.progress_logger import init_progress_logger, get_progress_logger
from ..config import Config

# Global cost meter
COST = CostMeter()


async def main():
    """BL3 experiment main function"""
    # Initialize progress logger
    logger = init_progress_logger("bl3")
    logger.log("=== Starting BL3 Experiment ===", print_to_console=False)
    
    # Set seed
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    
    # 1. Data preparation
    logger.log("Loading FEMNIST dataset...", print_to_console=False)
    client_train_ds, client_test_ds, num_classes = setup_femnist_by_writer(
        Config.NUM_CLIENTS
    )
    logger.log_complete("Data loading")
    global_test_loader = DataLoader(
        ConcatDataset(client_test_ds),
        batch_size=512
    )
    print(f"Total global test samples: {len(global_test_loader.dataset)}", flush=True)
    
    # Create test loader per client (for Personalized Accuracy measurement)
    client_test_loaders = [
        DataLoader(test_ds, batch_size=512, shuffle=False)
        for test_ds in client_test_ds
    ]
    
    # 2. Model initialization
    model = SimpleCNN(num_classes).to(Config.DEVICE)
    
    # Store models per cluster (client ID -> cluster ID mapping)
    client_to_cluster = {}
    cluster_models_dict = {}  # cluster ID -> model state_dict
    
    # 3. Build network topology
    uav_links = [
        Link(f"UAV{i}-SAT", 30+i*5, 10, Config.UAV_SAT_BW, 0.01)
        for i in range(Config.NUM_UAV)
    ]
    cli_links = [
        Link(f"C->UAV{i}", 50, 25, Config.CLIENT_UPLINK_BW, 0.02)
        for i in range(Config.NUM_UAV)
    ]
    
    uav_qs = [asyncio.Queue() for _ in range(Config.NUM_UAV)]
    sat_q = asyncio.Queue()
    
    sat = SatAgg(sat_q)
    uavs = [
        UAV(f"UAV{i}", uav_links[i], uav_qs[i], sat_q)
        for i in range(Config.NUM_UAV)
    ]
    
    # 4. Create clients
    clients = []
    for i in range(Config.NUM_CLIENTS):
        uav_idx = i % Config.NUM_UAV
        clients.append(
            FLClient(
                f"C{i}",
                f"UAV{uav_idx}",
                cli_links[uav_idx],
                DataLoader(
                    client_train_ds[i],
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True
                )
            )
        )
    
    # 5. Start background tasks
    bg_tasks = [asyncio.create_task(x.run()) for x in uavs + [sat]]
    
    print(f"\n=== Starting BL3 (Clustering Only) Simulation (Goal: GA {Config.CLUSTERING_TARGET_ACC}%) ===", flush=True)
    
    # 6. FL round execution
    for r in range(Config.ROUNDS):
        print(f"\n=== Round {r+1} ===", flush=True)
        
        # [BL3] Random client selection (no DCS)
        selected = random.sample(
            clients,
            int(Config.NUM_CLIENTS * Config.SAMPLE_FRAC)
        )
        
        # Client training and transmission
        async def client_task(c):
            # Use cluster model if available, otherwise use global model
            if c.id in client_to_cluster:
                cluster_id = client_to_cluster[c.id]
                if cluster_id in cluster_models_dict:
                    cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
                    cluster_model.load_state_dict(cluster_models_dict[cluster_id])
                    sd = c.train(cluster_model)
                else:
                    sd = c.train(model)
            else:
                sd = c.train(model)
            
            payload = get_state_dict_bytes(sd)
            
            if sent := await c.link.transmit(payload, cost_meter=COST):
                await uav_qs[int(c.uav_id[-1])].put(
                    Packet(c.id, c.uav_id, time.time(), sent)
                )
        
        await asyncio.gather(*(client_task(c) for c in selected))
        await asyncio.sleep(1)
        
        # [BL3 Core] Clustering and aggregation
        if sat.buffer:
            print(f"  Received {len(sat.buffer)} updates.", flush=True)
            logger.log(f"Round {r+1}: Received {len(sat.buffer)} updates", print_to_console=False)
            
            # sat.buffer is list of (client_id, state_dict) tuples
            # Separate state_dict list and client_id list
            buffer_state_dicts = [sd for _, sd in sat.buffer]
            buffer_client_ids = [cid for cid, _ in sat.buffer]
            
            # Calculate client data size weights
            client_weights = []
            client_id_to_idx = {c.id: i for i, c in enumerate(clients)}
            for client_id in buffer_client_ids:
                if client_id and client_id in client_id_to_idx:
                    client_idx = client_id_to_idx[client_id]
                    # Client dataset size
                    data_size = len(client_train_ds[client_idx])
                    client_weights.append(float(data_size))
                else:
                    # Uniform weight if client_id missing or not matched
                    client_weights.append(1.0)
            
            # Model similarity-based clustering
            print("  Computing model similarity and clustering...")
            logger.log(f"Round {r+1}: Computing model similarity and clustering...", print_to_console=False)
            similarity_matrix = compute_similarity_matrix(buffer_state_dicts)
            
            # Auto-determine number of clusters (simple heuristic)
            num_clusters = max(2, min(5, len(buffer_state_dicts) // 3))
            cluster_labels = cluster_models(buffer_state_dicts, num_clusters=num_clusters)
            
            print(f"  Clustered into {len(set(cluster_labels))} clusters: {dict(zip(range(len(cluster_labels)), cluster_labels))}", flush=True)
            
            # Aggregate models per cluster (using data size weights)
            new_cluster_models = cluster_based_aggregation(
                buffer_state_dicts, 
                cluster_labels,
                weights=client_weights
            )
            
            # Update cluster assignment for successfully transmitted clients (exact matching)
            # Match buffer_client_ids and cluster_labels
            for idx, client_id in enumerate(buffer_client_ids):
                if client_id and idx < len(cluster_labels):
                    client_to_cluster[client_id] = cluster_labels[idx]
            
            # Update cluster model (maintain and improve)
            # Use cluster size-based weights
            for cluster_id, cluster_model_state in new_cluster_models.items():
                if cluster_id in cluster_models_dict:
                    # Calculate current round cluster size
                    current_cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
                    # Weight of existing cluster model (considering cumulative effect of previous rounds)
                    # Simple weighted average with 1:1 ratio to current round size
                    cluster_models_dict[cluster_id] = fedavg(
                        [cluster_models_dict[cluster_id], cluster_model_state],
                        weights=[1.0, float(current_cluster_size)]
                    )
                else:
                    # Add new cluster model
                    cluster_models_dict[cluster_id] = cluster_model_state
            
            print(f"  Updated {len(cluster_models_dict)} cluster models.", flush=True)
            
            # Keep global model only for auxiliary metrics (average of all cluster models)
            if cluster_models_dict:
                all_cluster_models = list(cluster_models_dict.values())
                # Cluster size-based weights (number of clients in each cluster)
                cluster_weights = []
                for cluster_id in cluster_models_dict.keys():
                    cluster_size = sum(1 for cid in client_to_cluster.values() if cid == cluster_id)
                    cluster_weights.append(float(max(1, cluster_size)))  # Minimum 1
                model.load_state_dict(fedavg(all_cluster_models, weights=cluster_weights))
            else:
                model.load_state_dict(fedavg(buffer_state_dicts, weights=client_weights))
            
            sat.buffer.clear()
        else:
            print("  No updates received this round.", flush=True)
        
        # Evaluation: Global Accuracy
        logger.log(f"Round {r+1}: Evaluating model...", print_to_console=False)
        model.eval()
        
        # Show evaluation progress (logged to progress log only)
        try:
            from tqdm import tqdm
            import sys
            if logger:
                tqdm_file = open(logger.log_file, 'a')
                test_iter = tqdm(global_test_loader, desc="Evaluating", file=tqdm_file)
            else:
                test_iter = tqdm(global_test_loader, desc="Evaluating")
        except ImportError:
            test_iter = global_test_loader
            tqdm_file = None
        
        try:
            total_loss = 0.0
            correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for x, y in test_iter:
                    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                    outputs = model(x)
                    loss = F.cross_entropy(outputs, y, reduction='sum')
                    total_loss += loss.item()
                    pred = outputs.argmax(1)
                    correct += (pred == y).sum().item()
                    total_samples += y.size(0)
        finally:
            if 'tqdm_file' in locals() and tqdm_file:
                tqdm_file.close()
                import sys
                sys.stdout = sys.__stdout__
        
        ga = correct / total_samples * 100.0
        gl = total_loss / total_samples  # Global Loss
        
        # Measure Personalized Accuracy (evaluate cluster model on each client local test set)
        personalized_accs = []
        personalized_losses = []
        
        if cluster_models_dict and client_to_cluster:
            for client_idx, client in enumerate(clients):
                if client.id in client_to_cluster:
                    cluster_id = client_to_cluster[client.id]
                    if cluster_id in cluster_models_dict:
                        # Evaluate with cluster model
                        cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
                        cluster_model.load_state_dict(cluster_models_dict[cluster_id])
                        cluster_model.eval()
                        
                        test_loader = client_test_loaders[client_idx]
                        correct_local = 0
                        total_local = 0
                        loss_local = 0.0
                        
                        with torch.no_grad():
                            for x, y in test_loader:
                                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                                outputs = cluster_model(x)
                                loss = F.cross_entropy(outputs, y, reduction='sum')
                                loss_local += loss.item()
                                pred = outputs.argmax(1)
                                correct_local += (pred == y).sum().item()
                                total_local += y.size(0)
                        
                        if total_local > 0:
                            pa = correct_local / total_local * 100.0
                            pl = loss_local / total_local
                            personalized_accs.append(pa)
                            personalized_losses.append(pl)
        
        # Calculate Personalized Accuracy average
        pa_avg = sum(personalized_accs) / len(personalized_accs) if personalized_accs else 0.0
        pl_avg = sum(personalized_losses) / len(personalized_losses) if personalized_losses else 0.0
        
        logger.log(f"Round {r+1}: Global Accuracy = {ga:.2f}%, Global Loss = {gl:.4f} | "
                  f"Personalized Accuracy = {pa_avg:.2f}%, Personalized Loss = {pl_avg:.4f}", 
                  print_to_console=False)
        
        # Efficiency measurement
        r_bytes, r_delay = COST.end_round()
        
        print(f"  [Perf] Global GA: {ga:.2f}% | Global Loss: {gl:.4f}", flush=True)
        print(f"  [Perf] Personalized PA: {pa_avg:.2f}% | Personalized Loss: {pl_avg:.4f} "
              f"({len(personalized_accs)} clients)", flush=True)
        print(f"  [Effi] Round Cost: {r_bytes/1024:.0f} KB, +{r_delay:.2f}s simulated time", flush=True)
        print(f"  [Cumul] Total Data: {COST.total_cum_bytes/1024/1024:.2f} MB | "
              f"Total Time: {COST.total_cum_time:.2f}s", flush=True)
        
        # Check target achievement (based on Personalized Accuracy)
        if pa_avg >= Config.CLUSTERING_TARGET_ACC:
            print(f"\n!!! Target Personalized Accuracy ({Config.CLUSTERING_TARGET_ACC}%) Reached at Round {r+1} !!!", flush=True)
            print(f"FINAL Metrics -> Personalized PA: {pa_avg:.2f}% | Personalized Loss: {pl_avg:.4f} | "
                  f"Global GA: {ga:.2f}% | Global Loss: {gl:.4f} | "
                  f"Comm Cost: {COST.total_cum_bytes/1024/1024:.2f} MB | "
                  f"Time Cost: {COST.total_cum_time:.2f}s", flush=True)
            break
    
    # Save experiment results summary
    results_dir = Path("results/bl3")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"bl3_summary_{timestamp}.txt"
    
    with open(result_file, 'w') as f:
        f.write(f"=== BL3 Experiment Summary ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Rounds: {r+1}\n")
        f.write(f"\n--- Final Performance Metrics ---\n")
        f.write(f"Global Accuracy: {ga:.2f}%\n")
        f.write(f"Global Loss: {gl:.4f}\n")
        f.write(f"Personalized Accuracy: {pa_avg:.2f}%\n")
        f.write(f"Personalized Loss: {pl_avg:.4f}\n")
        f.write(f"\n--- Efficiency Metrics ---\n")
        f.write(f"Total Communication Cost: {COST.total_cum_bytes/1024/1024:.2f} MB\n")
        f.write(f"Total Time Cost: {COST.total_cum_time:.2f} seconds\n")
        f.write(f"Average Round Cost: {COST.total_cum_bytes/1024/1024/(r+1):.2f} MB\n")
        f.write(f"Average Round Time: {COST.total_cum_time/(r+1):.2f} seconds\n")
    
    print(f"\nResults saved to: {result_file}", flush=True)
    
    # Cleanup
    for t in bg_tasks:
        t.cancel()


if __name__ == "__main__":
    asyncio.run(main())

