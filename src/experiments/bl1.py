"""
BL1: Hierarchical FedAvg (Basic hierarchical federated learning)
- Random client selection
- UAV performs local FedAvg (cluster 0 only, no clustering)
- Satellite performs global FedAvg
- No DCS or clustering
"""
import asyncio
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ..core import (
    setup_femnist_by_writer, SimpleCNN, Link, CostMeter,
    FLClient, UAV, SatAgg, Packet, fedavg
)
from ..utils import get_state_dict_bytes, bytes_to_state_dict
from ..utils.progress_logger import init_progress_logger, get_progress_logger
from ..utils.cluster_logger import log_cluster_assignments
from ..config import Config

# Global cost meter
COST = CostMeter()


async def main():
    """BL1 experiment main function"""
    # Initialize progress logger
    logger = init_progress_logger("bl1")
    logger.log("=== Starting BL1 Experiment ===", print_to_console=False)
    
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
        
        # Assign all clients to cluster 0 (no clustering in BL1)
        uavs[uav_idx].client_to_cluster[client_id] = 0
        client.cluster_id = 0
    
    # 5. Start background tasks
    bg_tasks = [asyncio.create_task(x.run()) for x in uavs + [sat]]
    
    print(f"\n=== Starting BL1 Simulation (Goal: Avg Test Acc {Config.TARGET_ACC}%) ===", flush=True)
    
    # 6. FL round execution
    for r in range(Config.ROUNDS):
        print(f"\n=== Round {r+1} ===", flush=True)
        
        # Reset UAV buffers
        for uav in uavs:
            uav.reset_round_buffer()
        
        # [BL1] Random client selection
        selected = random.sample(
            clients,
            int(Config.NUM_CLIENTS * Config.SAMPLE_FRAC)
        )
        
        # Client training and transmission
        async def client_task(c):
            sd = c.train(model)
            payload = get_state_dict_bytes(sd)
            
            if sent := await c.link.transmit(payload, cost_meter=COST):
                await uav_qs[int(c.uav_id.split('_')[1])].put(
                    Packet(c.id, c.uav_id, time.time(), sent)
                )
        
        await asyncio.gather(*(client_task(c) for c in selected))
        await asyncio.sleep(0.5)  # Wait for packets to arrive at UAVs
        
        # Process client updates at UAVs
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
                except asyncio.QueueEmpty:
                    break
            
            # Aggregate cluster models (only cluster 0 for BL1)
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
        
        # Satellite aggregation
        if sat.buffer:
            # sat.buffer contains UAV payloads with cluster models
            # For BL1, all UAVs send cluster 0 models
            cluster_0_models = []
            cluster_0_weights = []
            
            for uav_payload in sat.buffer:
                for cluster_info in uav_payload.get("clusters", []):
                    if cluster_info["cluster_id"] == 0:
                        cluster_0_models.append(cluster_info["state_dict"])
                        cluster_0_weights.append(cluster_info["data_weight"])
            
            if cluster_0_models:
                model.load_state_dict(fedavg(cluster_0_models, weights=cluster_0_weights))
                print(f"  Aggregated {len(cluster_0_models)} UAV updates.")
            
            sat.buffer.clear()
        else:
            print("  No updates received this round.")

        # Log cluster assignments for selected clients
        if selected:
            log_cluster_assignments("bl1", r + 1, selected, sat.client_to_cluster)
        
        # Evaluation: Per-client test accuracy
        logger.log(f"Round {r+1}: Evaluating model...", print_to_console=False)
        model.eval()
        
        client_accuracies = []
        client_losses = []
        
        for client_idx, client in enumerate(clients):
            test_loader = client_test_loaders[client_idx]
            correct = 0
            total = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                    outputs = model(x)
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
        if avg_acc >= Config.TARGET_ACC:
            print(f"\n!!! Target Accuracy ({Config.TARGET_ACC}%) Reached at Round {r+1} !!!", flush=True)
            print(f"FINAL Metrics -> Avg Test Acc: {avg_acc:.2f}% | Avg Loss: {avg_loss:.4f} | "
                  f"Comm Cost: {COST.total_cum_bytes/1024/1024:.2f} MB | "
                  f"Time Cost: {COST.total_cum_time:.2f}s", flush=True)
            break
    
    # Save experiment results summary
    results_dir = Path("results/bl1")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"bl1_summary_{timestamp}.txt"
    
    with open(result_file, 'w') as f:
        f.write(f"=== BL1 Experiment Summary ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Rounds: {r+1}\n")
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
