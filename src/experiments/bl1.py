"""
BL1: Hierarchical FedAvg (기본 계층적 연합학습)
- 랜덤 클라이언트 선택
- 단순 평균 집계
- DCS 및 클러스터링 미적용
"""
import asyncio
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from ..core import (
    setup_femnist_by_writer, SimpleCNN, Link, CostMeter,
    FLClient, UAV, SatAgg, Packet, fedavg
)
from ..utils import get_state_dict_bytes
from ..utils.progress_logger import init_progress_logger, get_progress_logger
from ..config import Config

# 전역 비용 측정기
COST = CostMeter()


async def main():
    """BL1 실험 메인 함수"""
    # 진행 상황 로거 초기화
    logger = init_progress_logger("bl1")
    logger.log("=== Starting BL1 Experiment ===", print_to_console=False)
    
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
    print(f"Total global test samples: {len(global_test_loader.dataset)}")
    
    # 2. 모델 초기화
    model = SimpleCNN(num_classes).to(Config.DEVICE)
    
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
    
    print(f"\n=== Starting BL1 Simulation (Goal: GA {Config.TARGET_ACC}%) ===", flush=True)
    
    # 6. FL 라운드 실행
    for r in range(Config.ROUNDS):
        print(f"\n=== Round {r+1} ===", flush=True)
        
        # [BL1] 랜덤 클라이언트 선택
        selected = random.sample(
            clients,
            int(Config.NUM_CLIENTS * Config.SAMPLE_FRAC)
        )
        
        # 클라이언트 학습 및 전송
        async def client_task(c):
            sd = c.train(model)
            payload = get_state_dict_bytes(sd)
            
            # Link.transmit에 cost_meter 전달
            if sent := await c.link.transmit(payload, cost_meter=COST):
                await uav_qs[int(c.uav_id[-1])].put(
                    Packet(c.id, c.uav_id, time.time(), sent)
                )
        
        await asyncio.gather(*(client_task(c) for c in selected))
        await asyncio.sleep(1)  # UAV가 위성으로 전송할 시간
        
        # 집계
        if sat.buffer:
            # sat.buffer는 (client_id, state_dict) 튜플 리스트
            buffer_state_dicts = [sd for _, sd in sat.buffer]
            # 클라이언트 데이터 크기 가중치 계산
            client_weights = []
            client_id_to_idx = {c.id: i for i, c in enumerate(clients)}
            for client_id, _ in sat.buffer:
                if client_id and client_id in client_id_to_idx:
                    client_idx = client_id_to_idx[client_id]
                    data_size = len(client_train_ds[client_idx])
                    client_weights.append(float(data_size))
                else:
                    client_weights.append(1.0)
            
            model.load_state_dict(fedavg(buffer_state_dicts, weights=client_weights))
            print(f"  Aggregated {len(sat.buffer)} updates.")
            sat.buffer.clear()
        else:
            print("  No updates received this round.")
        
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
        logger.log(f"Round {r+1}: Global Accuracy = {ga:.2f}%, Global Loss = {gl:.4f}", print_to_console=False)
        
        # 효율성 측정
        r_bytes, r_delay = COST.end_round()
        
        print(f"  [Perf] Global GA: {ga:.2f}% | Global Loss: {gl:.4f}", flush=True)
        print(f"  [Effi] Round Cost: {r_bytes/1024:.0f} KB, +{r_delay:.2f}s simulated time", flush=True)
        print(f"  [Cumul] Total Data: {COST.total_cum_bytes/1024/1024:.2f} MB | "
              f"Total Time: {COST.total_cum_time:.2f}s", flush=True)
        
        # 목표 달성 체크
        if ga >= Config.TARGET_ACC:
            print(f"\n!!! Target Accuracy ({Config.TARGET_ACC}%) Reached at Round {r+1} !!!", flush=True)
            print(f"FINAL Metrics -> Accuracy: {ga:.2f}% | Loss: {gl:.4f} | "
                  f"Comm Cost: {COST.total_cum_bytes/1024/1024:.2f} MB | "
                  f"Time Cost: {COST.total_cum_time:.2f}s", flush=True)
            break
    
    # 정리
    for t in bg_tasks:
        t.cancel()


if __name__ == "__main__":
    asyncio.run(main())

