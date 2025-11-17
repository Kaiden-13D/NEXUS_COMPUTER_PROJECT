"""
BL3: Clustering Only
- Random client selection
- UAV performs local FedAvg per cluster
- Satellite performs clustering and manages global cluster models
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
from ..utils import get_state_dict_bytes, bytes_to_state_dict
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
    
    # Create test loaders per client (for per-client accuracy)
    client_test_loaders = [
        DataLoader(test_ds, batch_size=512, shuffle=False)
        for test_ds in client_test_ds
    ]
    
    # 2. Model initialization
    model = SimpleCNN(num_classes).to(Config.DEVICE)
    
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
        UAV(f"uav_{i}", uav_links[i], uav_qs[i], sat_q)
        for i in range(Config.NUM_UAV)
    ]
    
    # 4. Create clients and assign to UAVs
    clients = []
    client_id_to_idx = {}
    for i in range(Config.NUM_CLIENTS):
        uav_idx = i % Config.NUM_UAV
        client_id = f"client_{i}"
        client_id_to_idx[client_id] = i
        
        client = FLClient(
            id=client_id,
            uav_id=f"uav_{uav_idx}",
            link=cli_links[uav_idx],
            train_loader=DataLoader(
                client_train_ds[i],
                batch_size=Config.BATCH_SIZE,
                shuffle=True
            )
        )
        clients.append(client)
        uavs[uav_idx].assigned_clients.append(client)
        
        # Initially, all clients assigned to cluster 0 (will be updated after round 0)
        uavs[uav_idx].client_to_cluster[client_id] = 0
        client.cluster_id = 0
    
    # 5. Start background tasks
    bg_tasks = [asyncio.create_task(x.run()) for x in uavs + [sat]]
    
    print(f"\n=== Starting BL3 (Clustering Only) Simulation (Goal: Avg Test Acc {Config.CLUSTERING_TARGET_ACC}%) ===", flush=True)
    
    # 6. FL round execution
    for r in range(Config.ROUNDS):
        print(f"\n=== Round {r+1} ===", flush=True)
        
        # Reset UAV buffers
        for uav in uavs:
            uav.reset_round_buffer()
        
        # [BL3] Random client selection (no DCS)
        selected = random.sample(
            clients,
            int(Config.NUM_CLIENTS * Config.SAMPLE_FRAC)
        )
        
        # Client training and transmission
        async def client_task(c):
            # Use cluster model if available, otherwise use global model
            cluster_id = c.cluster_id if c.cluster_id is not None else 0
            if cluster_id in sat.cluster_models:
                cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
                cluster_model.load_state_dict(sat.cluster_models[cluster_id])
                sd = c.train(cluster_model)
            else:
                sd = c.train(model)
            
            payload = get_state_dict_bytes(sd)
            
            if sent := await c.link.transmit(payload, cost_meter=COST):
                uav_idx = int(c.uav_id.split('_')[1])
                await uav_qs[uav_idx].put(
                    Packet(c.id, c.uav_id, time.time(), sent)
                )
        
        await asyncio.gather(*(client_task(c) for c in selected))
        await asyncio.sleep(0.5)  # Wait for packets to arrive at UAVs
        
        # Process client updates at UAVs
        # Store individual client models for round 0 clustering
        individual_client_models = {}  # {client_id: (state_dict, data_size)}
        
        for uav in uavs:
            # Collect updates from queue
            while not uav.in_q.empty():
                try:
                    pkt = uav.in_q.get_nowait()
                    if isinstance(pkt, Packet):
                        client_id = pkt.src
                        state_dict = bytes_to_state_dict(pkt.data)
                        # Get data size for weight
                        if client_id in client_id_to_idx:
                            client_idx = client_id_to_idx[client_id]
                            data_size = len(client_train_ds[client_idx])
                        else:
                            data_size = 1.0
                        uav.add_client_update(client_id, state_dict, float(data_size))
                        # Store for round 0 clustering
                        if r == 0:
                            individual_client_models[client_id] = (state_dict, data_size)
                except asyncio.QueueEmpty:
                    break
            
            # Aggregate cluster models
            cluster_models = uav.aggregate_cluster_models()
            if cluster_models:
                # Send to satellite
                client_id_to_data_size = {
                    c.id: len(client_train_ds[client_id_to_idx[c.id]])
                    for c in uav.assigned_clients
                }
                await uav.send_cluster_models_to_satellite(
                    cluster_models, client_id_to_data_size, cost_meter=COST
                )
        
        await asyncio.sleep(0.5)  # Wait for UAVs to send to satellite
        
        # Satellite processing: aggregate cluster models and perform clustering
        if sat.buffer:
            print(f"  Received {len(sat.buffer)} UAV updates.", flush=True)
            logger.log(f"Round {r+1}: Received {len(sat.buffer)} UAV updates", print_to_console=False)
            
            # Collect all cluster models from UAVs
            # Structure: {cluster_id: [(state_dict, weight, client_ids), ...]}
            cluster_collections = defaultdict(list)
            all_client_ids_in_round = []
            
            for uav_payload in sat.buffer:
                for cluster_info in uav_payload.get("clusters", []):
                    cluster_id = cluster_info["cluster_id"]
                    state_dict = cluster_info["state_dict"]
                    data_weight = cluster_info["data_weight"]
                    client_ids = cluster_info.get("client_ids", [])
                    
                    cluster_collections[cluster_id].append((state_dict, data_weight, client_ids))
                    all_client_ids_in_round.extend(client_ids)
            
            # Round 0: Determine number of clusters using individual client models
            if r == 0 and sat.num_clusters is None:
                print("  Round 0: Determining number of clusters...", flush=True)
                
                # Use individual client models for clustering (more accurate)
                if len(individual_client_models) > 1:
                    all_state_dicts = []
                    all_client_ids_list = []
                    all_weights = []
                    
                    for client_id, (state_dict, data_size) in individual_client_models.items():
                        all_state_dicts.append(state_dict)
                        all_client_ids_list.append(client_id)
                        all_weights.append(data_size)
                    
                    # Perform clustering on individual client models
                    similarity_matrix = compute_similarity_matrix(all_state_dicts)
                    
                    if Config.AUTO_DETERMINE_CLUSTERS:
                        # Auto-determine number of clusters
                        num_clusters = max(2, min(5, len(all_state_dicts) // 3))
                    else:
                        num_clusters = Config.NUM_CLUSTERS
                    
                    cluster_labels = cluster_models(all_state_dicts, num_clusters=num_clusters)
                    sat.num_clusters = num_clusters
                    
                    print(f"  Determined {num_clusters} clusters from {len(all_state_dicts)} client models.", flush=True)
                    
                    # Map client IDs to clusters
                    new_client_to_cluster = {}
                    for idx, client_id in enumerate(all_client_ids_list):
                        new_client_to_cluster[client_id] = cluster_labels[idx]
                    
                    # Update satellite's client_to_cluster
                    sat.client_to_cluster.update(new_client_to_cluster)
                else:
                    sat.num_clusters = Config.NUM_CLUSTERS
                    print(f"  Using default {sat.num_clusters} clusters.", flush=True)
            
            # Aggregate cluster models per cluster (global aggregation)
            num_clusters = sat.num_clusters if sat.num_clusters else Config.NUM_CLUSTERS
            
            # For each cluster, aggregate all UAV cluster models
            for cluster_id in range(num_clusters):
                # Collect all state_dicts and weights for this cluster
                cluster_state_dicts = []
                cluster_weights = []
                
                # Check if any UAV sent models for this cluster
                if cluster_id in cluster_collections:
                    for state_dict, weight, client_ids in cluster_collections[cluster_id]:
                        cluster_state_dicts.append(state_dict)
                        cluster_weights.append(weight)
                
                # Also check if we need to aggregate from previous rounds' cluster models
                if cluster_state_dicts:
                    # Aggregate UAV cluster models
                    aggregated_model = fedavg(cluster_state_dicts, weights=cluster_weights)
                    
                    # Update or create global cluster model
                    if cluster_id in sat.cluster_models:
                        # Weighted average with previous cluster model
                        total_weight = sum(cluster_weights)
                        sat.cluster_models[cluster_id] = fedavg(
                            [sat.cluster_models[cluster_id], aggregated_model],
                            weights=[1.0, total_weight]
                        )
                    else:
                        sat.cluster_models[cluster_id] = aggregated_model
            
            # For rounds >= 1, update cluster assignments if needed
            # For now, we keep the previous round's assignments
            # In a more sophisticated implementation, we could re-cluster based on
            # similarity to current cluster models
            
            # Distribute cluster assignments to UAVs
            for uav in uavs:
                # Get client IDs assigned to this UAV
                uav_client_mapping = {}
                for client in uav.assigned_clients:
                    if client.id in sat.client_to_cluster:
                        uav_client_mapping[client.id] = sat.client_to_cluster[client.id]
                
                if uav_client_mapping:
                    uav.update_client_cluster_mapping(uav_client_mapping)
            
            print(f"  Updated {len(sat.cluster_models)} global cluster models.", flush=True)
            sat.buffer.clear()
        else:
            print("  No updates received this round.", flush=True)
        
        # Evaluation: Per-client test accuracy using cluster models
        logger.log(f"Round {r+1}: Evaluating models...", print_to_console=False)
        
        client_accuracies = []
        client_losses = []
        
        for client_idx, client in enumerate(clients):
            # Determine which cluster model to use
            cluster_id = client.cluster_id if client.cluster_id is not None else 0
            if cluster_id in sat.cluster_models:
                eval_model = SimpleCNN(num_classes).to(Config.DEVICE)
                eval_model.load_state_dict(sat.cluster_models[cluster_id])
            else:
                eval_model = model
            
            eval_model.eval()
            test_loader = client_test_loaders[client_idx]
            correct = 0
            total = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                    outputs = eval_model(x)
                    loss = F.cross_entropy(outputs, y, reduction='sum')
                    total_loss += loss.item()
                    pred = outputs.argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            
            if total > 0:
                acc = correct / total * 100.0
                loss_val = total_loss / total
                client_accuracies.append(acc)
                client_losses.append(loss_val)
        
        avg_acc = sum(client_accuracies) / len(client_accuracies) if client_accuracies else 0.0
        avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
        
        logger.log(f"Round {r+1}: Avg Test Acc = {avg_acc:.2f}%, Avg Loss = {avg_loss:.4f}", print_to_console=False)
        
        # Efficiency measurement
        r_bytes, r_delay = COST.end_round()
        
        print(f"  [Perf] Avg Test Acc: {avg_acc:.2f}% | Avg Loss: {avg_loss:.4f} ({len(client_accuracies)} clients)", flush=True)
        print(f"  [Effi] Round Cost: {r_bytes/1024:.0f} KB, +{r_delay:.2f}s simulated time", flush=True)
        print(f"  [Cumul] Total Data: {COST.total_cum_bytes/1024/1024:.2f} MB | "
              f"Total Time: {COST.total_cum_time:.2f}s", flush=True)
        
        # Check target achievement
        if avg_acc >= Config.CLUSTERING_TARGET_ACC:
            print(f"\n!!! Target Accuracy ({Config.CLUSTERING_TARGET_ACC}%) Reached at Round {r+1} !!!", flush=True)
            print(f"FINAL Metrics -> Avg Test Acc: {avg_acc:.2f}% | Avg Loss: {avg_loss:.4f} | "
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
        f.write(f"Number of Clusters: {sat.num_clusters}\n")
        f.write(f"\n--- Final Performance Metrics ---\n")
        f.write(f"Average Test Accuracy: {avg_acc:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
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
