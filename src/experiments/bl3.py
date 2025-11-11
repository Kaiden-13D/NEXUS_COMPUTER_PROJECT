"""
BL3: Clustering Only (모델 유사도 기반 클러스터링만 적용)
- 랜덤 클라이언트 선택
- 모델 유사도 기반 클러스터링 적용
- 클러스터별 모델 집계
- DCS 미적용
- 데이터 이질성 완화 효과 측정 (정확도 향상)
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

# 전역 비용 측정기
COST = CostMeter()


async def main():
    """BL3 실험 메인 함수"""
    # 진행 상황 로거 초기화
    logger = init_progress_logger("bl3")
    logger.log("=== Starting BL3 Experiment ===", print_to_console=False)
    
    # 시드 설정
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    
    # 1. 데이터 준비
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
    
    # 클라이언트별 테스트 로더 생성 (Personalized Accuracy 측정용)
    client_test_loaders = [
        DataLoader(test_ds, batch_size=512, shuffle=False)
        for test_ds in client_test_ds
    ]
    
    # 2. 모델 초기화
    model = SimpleCNN(num_classes).to(Config.DEVICE)
    
    # 클러스터별 모델 저장 (클라이언트 ID -> 클러스터 ID 매핑)
    client_to_cluster = {}
    cluster_models_dict = {}  # 클러스터 ID -> 모델 state_dict
    
    # 3. 네트워크 토폴로지 구축
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
    
    # 4. 클라이언트 생성
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
    
    # 5. 백그라운드 태스크 시작
    bg_tasks = [asyncio.create_task(x.run()) for x in uavs + [sat]]
    
    print(f"\n=== Starting BL3 (Clustering Only) Simulation (Goal: GA {Config.CLUSTERING_TARGET_ACC}%) ===", flush=True)
    
    # 6. FL 라운드 실행
    for r in range(Config.ROUNDS):
        print(f"\n=== Round {r+1} ===", flush=True)
        
        # [BL3] 랜덤 클라이언트 선택 (DCS 미적용)
        selected = random.sample(
            clients,
            int(Config.NUM_CLIENTS * Config.SAMPLE_FRAC)
        )
        
        # 클라이언트 학습 및 전송
        async def client_task(c):
            # 클러스터 모델이 있으면 해당 모델 사용, 없으면 전역 모델 사용
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
        
        # [BL3 핵심] 클러스터링 및 집계
        if sat.buffer:
            print(f"  Received {len(sat.buffer)} updates.", flush=True)
            logger.log(f"Round {r+1}: Received {len(sat.buffer)} updates", print_to_console=False)
            
            # sat.buffer는 (client_id, state_dict) 튜플 리스트
            # state_dict 리스트와 client_id 리스트 분리
            buffer_state_dicts = [sd for _, sd in sat.buffer]
            buffer_client_ids = [cid for cid, _ in sat.buffer]
            
            # 클라이언트 데이터 크기 가중치 계산
            client_weights = []
            client_id_to_idx = {c.id: i for i, c in enumerate(clients)}
            for client_id in buffer_client_ids:
                if client_id and client_id in client_id_to_idx:
                    client_idx = client_id_to_idx[client_id]
                    # 클라이언트의 데이터셋 크기
                    data_size = len(client_train_ds[client_idx])
                    client_weights.append(float(data_size))
                else:
                    # client_id가 없거나 매칭되지 않으면 균등 가중치
                    client_weights.append(1.0)
            
            # 모델 유사도 기반 클러스터링
            print("  Computing model similarity and clustering...")
            logger.log(f"Round {r+1}: Computing model similarity and clustering...", print_to_console=False)
            similarity_matrix = compute_similarity_matrix(buffer_state_dicts)
            
            # 클러스터 수 자동 결정 (간단한 휴리스틱)
            num_clusters = max(2, min(5, len(buffer_state_dicts) // 3))
            cluster_labels = cluster_models(buffer_state_dicts, num_clusters=num_clusters)
            
            print(f"  Clustered into {len(set(cluster_labels))} clusters: {dict(zip(range(len(cluster_labels)), cluster_labels))}", flush=True)
            
            # 클러스터별 모델 집계 (데이터 크기 가중치 사용)
            new_cluster_models = cluster_based_aggregation(
                buffer_state_dicts, 
                cluster_labels,
                weights=client_weights
            )
            
            # 전송 성공한 클라이언트들의 클러스터 할당 업데이트 (정확한 매칭)
            # buffer_client_ids와 cluster_labels를 매칭
            for idx, client_id in enumerate(buffer_client_ids):
                if client_id and idx < len(cluster_labels):
                    client_to_cluster[client_id] = cluster_labels[idx]
            
            # 클러스터 모델 업데이트 (유지 및 개선)
            # 클러스터 크기 기반 가중치 사용
            for cluster_id, cluster_model_state in new_cluster_models.items():
                if cluster_id in cluster_models_dict:
                    # 현재 라운드의 클러스터 크기 계산
                    current_cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
                    # 기존 클러스터 모델의 가중치 (이전 라운드들의 누적 효과를 고려)
                    # 간단히 현재 라운드 크기와 1:1로 가중 평균
                    cluster_models_dict[cluster_id] = fedavg(
                        [cluster_models_dict[cluster_id], cluster_model_state],
                        weights=[1.0, float(current_cluster_size)]
                    )
                else:
                    # 새 클러스터 모델 추가
                    cluster_models_dict[cluster_id] = cluster_model_state
            
            print(f"  Updated {len(cluster_models_dict)} cluster models.", flush=True)
            
            # 전역 모델은 보조 지표용으로만 유지 (모든 클러스터 모델의 평균)
            if cluster_models_dict:
                all_cluster_models = list(cluster_models_dict.values())
                # 클러스터 크기 기반 가중치 (각 클러스터에 속한 클라이언트 수)
                cluster_weights = []
                for cluster_id in cluster_models_dict.keys():
                    cluster_size = sum(1 for cid in client_to_cluster.values() if cid == cluster_id)
                    cluster_weights.append(float(max(1, cluster_size)))  # 최소 1
                model.load_state_dict(fedavg(all_cluster_models, weights=cluster_weights))
            else:
                model.load_state_dict(fedavg(buffer_state_dicts, weights=client_weights))
            
            sat.buffer.clear()
        else:
            print("  No updates received this round.", flush=True)
        
        # 평가: Global Accuracy
        logger.log(f"Round {r+1}: Evaluating model...", print_to_console=False)
        model.eval()
        
        # 평가 진행 상황 표시 (progress 로그에만 기록)
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
        
        # Personalized Accuracy 측정 (각 클라이언트의 로컬 테스트셋에서 자신의 클러스터 모델 평가)
        personalized_accs = []
        personalized_losses = []
        
        if cluster_models_dict and client_to_cluster:
            for client_idx, client in enumerate(clients):
                if client.id in client_to_cluster:
                    cluster_id = client_to_cluster[client.id]
                    if cluster_id in cluster_models_dict:
                        # 클러스터 모델로 평가
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
        
        # Personalized Accuracy 평균 계산
        pa_avg = sum(personalized_accs) / len(personalized_accs) if personalized_accs else 0.0
        pl_avg = sum(personalized_losses) / len(personalized_losses) if personalized_losses else 0.0
        
        logger.log(f"Round {r+1}: Global Accuracy = {ga:.2f}%, Global Loss = {gl:.4f} | "
                  f"Personalized Accuracy = {pa_avg:.2f}%, Personalized Loss = {pl_avg:.4f}", 
                  print_to_console=False)
        
        # 효율성 측정
        r_bytes, r_delay = COST.end_round()
        
        print(f"  [Perf] Global GA: {ga:.2f}% | Global Loss: {gl:.4f}", flush=True)
        print(f"  [Perf] Personalized PA: {pa_avg:.2f}% | Personalized Loss: {pl_avg:.4f} "
              f"({len(personalized_accs)} clients)", flush=True)
        print(f"  [Effi] Round Cost: {r_bytes/1024:.0f} KB, +{r_delay:.2f}s simulated time", flush=True)
        print(f"  [Cumul] Total Data: {COST.total_cum_bytes/1024/1024:.2f} MB | "
              f"Total Time: {COST.total_cum_time:.2f}s", flush=True)
        
        # 목표 달성 체크 (Personalized Accuracy 기준)
        if pa_avg >= Config.CLUSTERING_TARGET_ACC:
            print(f"\n!!! Target Personalized Accuracy ({Config.CLUSTERING_TARGET_ACC}%) Reached at Round {r+1} !!!", flush=True)
            print(f"FINAL Metrics -> Personalized PA: {pa_avg:.2f}% | Personalized Loss: {pl_avg:.4f} | "
                  f"Global GA: {ga:.2f}% | Global Loss: {gl:.4f} | "
                  f"Comm Cost: {COST.total_cum_bytes/1024/1024:.2f} MB | "
                  f"Time Cost: {COST.total_cum_time:.2f}s", flush=True)
            break
    
    # 정리
    for t in bg_tasks:
        t.cancel()


if __name__ == "__main__":
    asyncio.run(main())

