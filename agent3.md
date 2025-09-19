1. 진행 상황 검토 및 핵심 연구 과제 확인
•초기 시뮬레이션 구현 완료: 사전에 3개의 클러스터를 가정한 뒤, 각 클러스터별로 
모델을 생성하여 정확도(Accuracy)를 측정하는 Baseline 실험 환경 구현을 완료했
음을 확인함.
 •핵심 연구 과제 명확화: 향후 연구의 핵심은 모델 간 유사도(Similarity)를 정의하고, 
이를 기반으로 클라이언트를 동적으로 군집화(Clustering)하는 알고리즘을 설계하는 
것임을 재확인함.
 2. 운영 시나리오 및 네트워크 모델 선행 설계
•선 시나리오 설계, 후 알고리즘 구체화: 유사도 측정 방식이라는 어려운 문제에 앞
서, 시스템의 전체적인 운영 흐름(Operational Flow)을 먼저 구체적으로 설계하는 
것이 중요함을 지도받음. 이 운영 시나리오가 향후 시뮬레이션과 알고리즘 개발의 
기반이 될 것임을 확인함.
 •네트워크 모델 정의: 운영 시나리오 설계 시 다음 항목들을 포함하여 명확히 정의
하기로 함.
 •구성 요소의 역할: 지상 클라이언트(IoT), UAV, 위성 등 각 계층의 명확한 역
할과 정보 흐름 정의.
 •파라미터 정의: 시뮬레이션에 사용될 Transmission Delay, CPU Time등 네
트워크 지표 정의.
 •중앙 서버(Control Server) 명세: 클러스터링에 필요한 정보를 취합하고 명령
을 내리는 중앙 서버의 위치(e.g., 위성) 및 정보 교환 주기 결정.
 3. 클러스터링 방법론 및 시뮬레이션 방향성 논의
•클러스터링 목표 재정의: 클러스터링의 목표는 각 클라이언트의 실제 소속 그룹을 
모르는 상태(Privacy-preserving)에서, 모델의 특성만으로 최적의 그룹을 형성하고 
개인화된 모델을 제공하는 것임을 논의함.
 •시뮬레이션 접근법: 초기 단계에서는 복잡도를 낮추기 위해, 클러스터의 개수(k)를 
하이퍼파라미터로 가정한 정적(Static) 데이터 환경에서 시작하는 것이 효율적이라
는 점에 대해 확인함.


전반적인 내용에 대해서 보완 설명 드리겠습니다.
전체 user를 Zone에 관련 없이 K개의 cluster로 나눈다고 생각하고, 따라서 K개의 global model을 만드는 것이 목표라고 생각하는 것이 좋겠습니다. 이를 구성하는 과정에서, 1) local model들을 aggregation하는 목적으로 satellite가 필요, 2) 모든 user들이 satellite에 연결되는 것이 비효율적이니, 중간 전달자인 UAV가 존재하며, 각 UAV는 각 Zone에 하나씩 존재한다 이렇게 생각하시는 것이 좋을 것 같습니다.

Q1. Zone에 대한 cluster는 존재하지 않는다고 생각하는 편이 좋을 것 같습니다. Zone을 나눈 이유는, 통신 link를 안정적으로 만들기 위해, 가까운 user들끼리 묶어서 하나의 UAV를 통해 satellite로 model을 relay하기 위함입니다.

Q2. 연구 목적 관점에서는, 다중 global model이 더 성능이 좋게 나오는 상황을 구성해야 합니다. (연구의 근본적인 목표입니다) real-world data는 non-iid 분포를 지니는 경우가 일반적이며, 단일 global model로는 이러한 상황에 적절하게 대처하기 어렵기 때문입니다. 

---
### Refactoring Implementation (2025-09-19)

To align the simulation with the research goal of creating multiple global models (Pattern 2), the `main.py` script was significantly refactored.

**Summary of Key Changes:**

1.  **Architecture Shift:** The core logic was changed from a "Single Global Model with Local Clustering" (Pattern 1) to a "Multiple Global Models with Global Clustering" (Pattern 2).

2.  **`Satellite` Class Refactoring:**
    *   The `Satellite` is now the central orchestrator of the entire clustering process.
    *   It initializes and maintains **K distinct global models**.
    *   A new method, `assign_clients_to_global_clusters`, was added to globally cluster all clients based on their loss against the K models.
    *   The `train_round` logic was rewritten to perform this global clustering first, followed by partitioned aggregation, where each global model is updated only by the clients in its cluster.
    *   The `test` method was updated to evaluate the performance of the K-model ensemble.

3.  **`UAV` Class Simplification:**
    *   The `UAV`'s role has been reduced to a simple communication relay.
    *   All methods and attributes related to local clustering (`assign_clients_to_clusters`, `cluster_models`, `train_zone`, etc.) have been removed.

4.  **Configuration and Execution:**
    *   The `Args` class was updated to use `num_global_clusters` (`K`) instead of `num_clusters_per_zone`.
    *   The main execution loop was updated to correctly initialize the `Satellite` and save the final K global models.