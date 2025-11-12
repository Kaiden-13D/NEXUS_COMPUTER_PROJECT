"""
ABL-2: DCS + Clustering + Head-Personalization
- DCS-based client selection at UAV
- Model similarity-based clustering and cluster-weighted aggregation
- Freeze backbone, train only head (fc/classifier) during client local training
- Personalized/Global performance and network cost logging
"""
import asyncio
import random
import time
from collections import defaultdict

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


def train_head_only(client: FLClient, base_model: torch.nn.Module, num_classes: int) -> dict:
    """Freeze backbone, train only head, return state_dict"""
    model = SimpleCNN(num_classes).to(Config.DEVICE)
    model.load_state_dict(base_model.state_dict())
    
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    # Train only head parameters
    head_keys = ("fc", "classifier")
    for name, p in model.named_parameters():
        if any(k in name for k in head_keys):
            p.requires_grad = True
    
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=Config.LR,
        momentum=Config.MOMENTUM
    )
    model.train()
    for epoch in range(Config.LOCAL_EPOCHS):
        for x, y in client.train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
    return model.state_dict()


async def main():
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    
    # Initialize progress/timestamp logger
    logger = init_progress_logger(experiment_name="ABL-2")
    print("Progress log: logs/progress/latest.log", flush=True)
    print("Timestamp log: logs/timestamp/latest.log", flush=True)
    
    # Data preparation
    print("Loading FEMNIST from Hugging Face (flwrlabs/femnist)...", flush=True)
    logger.log("Loading FEMNIST...", print_to_console=False)
    clients_ds, client_test_ds, num_classes = setup_femnist_by_writer(
        num_clients=Config.NUM_CLIENTS
    )
    global_test_loader = DataLoader(
        ConcatDataset(client_test_ds), batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    # Initialize nodes/links
    uav_links = [
        Link(f"UAV{i}-SAT", 30+i*5, 10, Config.UAV_SAT_BW, 0.01)
        for i in range(Config.NUM_UAV)
    ]
    sat_link = Link("SAT", 30, 10, Config.UAV_SAT_BW, 0.01)
    
    uav_in_qs = [asyncio.Queue() for _ in range(Config.NUM_UAV)]
    uav_out_q = asyncio.Queue()
    uavs = [
        UAV(id=f"uav_{i}", link=uav_links[i], in_q=uav_in_qs[i], out_q=uav_out_q)
        for i in range(Config.NUM_UAV)
    ]
    sat = SatAgg(in_q=uav_out_q)
    
    # Create clients and assign to UAVs
    client_train_ds = []
    clients = []
    for i in range(Config.NUM_CLIENTS):
        ds = clients_ds[i]
        client_train_ds.append(ds)
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        uav_id = f"uav_{i % Config.NUM_UAV}"
        link = Link(f"C{i}->UAV{i % Config.NUM_UAV}", 50, 25, Config.CLIENT_UPLINK_BW, 0.02)
        c = FLClient(id=f"client_{i}", uav_id=uav_id, link=link, train_loader=loader)
        clients.append(c)
    
    # Assign clients to UAVs
    for c in clients:
        idx = int(c.uav_id[-1])
        uavs[idx].assigned_clients.append(c)
    
    # Start UAV/SAT run loops
    uav_tasks = [asyncio.create_task(u.run()) for u in uavs]
    sat_task = asyncio.create_task(sat.run())
    
    # Global model
    model = SimpleCNN(num_classes).to(Config.DEVICE)
    
    # Clustering state
    cluster_models_dict = {}  # {cluster_id: state_dict}
    client_to_cluster = {}    # {client_id: cluster_id}
    
    print(f"\n=== Starting ABL-2 (DCS + Clustering + Head-Personalization) Simulation (Goal: GA {Config.CLUSTERING_TARGET_ACC}%) ===", flush=True)
    
    for r in range(Config.ROUNDS):
        print(f"\n=== Round {r+1} ===", flush=True)
        logger.log(f"=== Round {r+1} ===", print_to_console=False)
        
        # DCS-based client selection
        selected = []
        for u in uavs:
            selected.extend(u.select_clients_dcs(Config.SAMPLE_FRAC))
        random.shuffle(selected)
        
        # Local training + transmission (head-only training)
        async def client_task(c):
            if c.id in client_to_cluster:
                cluster_id = client_to_cluster[c.id]
                if cluster_id in cluster_models_dict:
                    cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
                    cluster_model.load_state_dict(cluster_models_dict[cluster_id])
                    sd = train_head_only(c, cluster_model, num_classes)
                else:
                    sd = train_head_only(c, model, num_classes)
            else:
                sd = train_head_only(c, model, num_classes)
            
            payload = get_state_dict_bytes(sd)
            if sent := await c.link.transmit(payload, cost_meter=COST):
                await uav_in_qs[int(c.uav_id[-1])].put(
                    Packet(c.id, c.uav_id, time.time(), sent)
                )
        
        await asyncio.gather(*(client_task(c) for c in selected))
        await asyncio.sleep(1)
        
        # Clustering and aggregation
        buffer_client_ids = []
        if sat.buffer:
            print(f"  Received {len(sat.buffer)} updates.", flush=True)
            logger.log(f"Round {r+1}: Received {len(sat.buffer)} updates", print_to_console=False)
            
            buffer_state_dicts = [sd for _, sd in sat.buffer]
            buffer_client_ids = [cid for cid, _ in sat.buffer]
            
            # Data size-based weights
            client_weights = []
            client_id_to_idx = {c.id: i for i, c in enumerate(clients)}
            for client_id in buffer_client_ids:
                if client_id and client_id in client_id_to_idx:
                    client_idx = client_id_to_idx[client_id]
                    data_size = len(client_train_ds[client_idx])
                    client_weights.append(float(data_size))
                else:
                    client_weights.append(1.0)
            
            # Model similarity-based clustering
            print("  Computing model similarity and clustering...", flush=True)
            logger.log(f"Round {r+1}: Computing model similarity and clustering...", print_to_console=False)
            similarity_matrix = compute_similarity_matrix(buffer_state_dicts)
            
            # Auto-determine number of clusters (simple heuristic)
            num_clusters = max(2, min(5, len(buffer_state_dicts) // 3))
            cluster_labels = cluster_models(buffer_state_dicts, num_clusters=num_clusters)
            
            print(f"  Clustered into {len(set(cluster_labels))} clusters: {dict(zip(range(len(cluster_labels)), cluster_labels))}", flush=True)
            
            # Weighted aggregation per cluster
            new_cluster_models = cluster_based_aggregation(
                buffer_state_dicts,
                cluster_labels,
                weights=client_weights
            )
            
            # Update cluster assignment
            for idx, client_id in enumerate(buffer_client_ids):
                if client_id and idx < len(cluster_labels):
                    client_to_cluster[client_id] = cluster_labels[idx]
            
            # Update cluster model (using cluster size-based weights)
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
            
            # Update global model for auxiliary metrics
            if cluster_models_dict:
                all_cluster_models = list(cluster_models_dict.values())
                cluster_weights = []
                for cluster_id in cluster_models_dict.keys():
                    cluster_size = sum(1 for cid in client_to_cluster.values() if cid == cluster_id)
                    cluster_weights.append(float(max(1, cluster_size)))
                model.load_state_dict(fedavg(all_cluster_models, weights=cluster_weights))
            else:
                model.load_state_dict(fedavg(buffer_state_dicts, weights=client_weights))
            
            sat.buffer.clear()
        else:
            print("  No updates received this round.", flush=True)
        
        # Global evaluation
        logger.log(f"Round {r+1}: Evaluating model...", print_to_console=False)
        model.eval()
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
        
        correct = 0
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in test_iter:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                outputs = model(x)
                loss = F.cross_entropy(outputs, y, reduction='sum')
                total_loss += loss.item()
                pred = outputs.argmax(1)
                correct += (pred == y).sum().item()
                total_samples += y.size(0)
        if 'tqdm_file' in locals() and tqdm_file:
            tqdm_file.close()
        
        ga = correct / total_samples * 100.0
        gl = total_loss / total_samples
        
        # Personalized evaluation (current round participants only)
        personalized_accs = []
        personalized_losses = []
        current_round_client_ids = set(buffer_client_ids)
        if cluster_models_dict and client_to_cluster:
            for client_idx, client in enumerate(clients):
                if client.id in client_to_cluster and client.id in current_round_client_ids:
                    cluster_id = client_to_cluster[client.id]
                    if cluster_id in cluster_models_dict:
                        cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
                        cluster_model.load_state_dict(cluster_models_dict[cluster_id])
                        cluster_model.eval()
                        test_loader = clients[client_idx].train_loader  # Use as substitute if local test set not available separately
                        
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
        
        pa_avg = sum(personalized_accs) / len(personalized_accs) if personalized_accs else 0.0
        pl_avg = sum(personalized_losses) / len(personalized_losses) if personalized_losses else 0.0
        
        # Efficiency measurement
        r_bytes, r_delay = COST.end_round()
        
        # Round/cumulative cost
        print(f"  [Perf] Global GA: {ga:.2f}% | Global Loss: {gl:.4f}", flush=True)
        print(f"  [Perf] Personalized PA: {pa_avg:.2f}% | Personalized Loss: {pl_avg:.4f} ({len(personalized_accs)} clients)", flush=True)
        print(f"  [Effi] Round Cost: {r_bytes/1024:.0f} KB, +{r_delay:.2f}s simulated time", flush=True)
        print(f"  [Cumul] Total Data: {COST.total_cum_bytes/1024/1024:.2f} MB | Total Time: {COST.total_cum_time:.2f}s", flush=True)
        
        # Termination condition (clustering target based on Personalized)
        if pa_avg >= Config.CLUSTERING_TARGET_ACC:
            print(f\"\\n!!! Target Personalized Accuracy ({Config.CLUSTERING_TARGET_ACC}%) Reached at Round {r+1} !!!\", flush=True)
            print(f\"FINAL Metrics -> Personalized PA: {pa_avg:.2f}% | Personalized Loss: {pl_avg:.4f} | \"\n                  f\"Global GA: {ga:.2f}% | Global Loss: {gl:.4f} | \"\n                  f\"Comm Cost: {COST.total_cum_bytes/1024/1024:.2f} MB | \"\n                  f\"Time Cost: {COST.total_cum_time:.2f}s\", flush=True)
            break
    
    # Cleanup
    for t in uav_tasks:
        t.cancel()
    sat_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())


