# Agent Blueprint: Zone + Clustering PFL (UE–UAV–Satellite)

## 0) Purpose & Scope
A distilled, implementation‑ready blueprint for a **zone + clustering** Personalized Federated Learning (PFL) system over a **three‑tier hierarchy**: **UE → UAV (edge) → Satellite (core)**. This document focuses on a **normalized scenario** (stable links, one task, unified model) while noting where extensions fit.

---

## 1) Core Ideas
- **Federated Learning (FL):** Train collaboratively without sharing raw data.
- **Personalized FL (PFL):** Optimize **per‑client** performance under non‑IID data.
- **Zones:** Administrative/physical partitions (here, a **zone ≈ one UAV domain**).
- **Clustering:** Group clients **within a zone** by similarity (update vectors, losses, features) to aggregate more homogeneously.
- **Hierarchy:** Two aggregation layers beyond clients: **UAV (zone) → Satellite (global)**.

---

## 2) Normalized Scenario Assumptions
- **Task/model:** Single task (e.g., classification) with one shared **backbone**; optional **personal heads**.
- **Data heterogeneity:** Non‑IID across UEs; moderately IID within clusters after grouping.
- **Connectivity:** UE↔UAV is moderate/steady; UAV↔Satellite is slower but reliable. Intermittent UE availability is allowed.
- **Compute:** UE (low), UAV (medium), Satellite (high).
- **Privacy & Security:** No raw data leaves UEs. (Optional) secure aggregation, DP noise, compression.
- **Timing:** Support both **synchronous rounds** and **asynchronous streaming** (detailed below).

---

## 3) Roles & Terminology
- **UE (client):** Trains locally E epochs/steps on private data.
- **UAV (zone server/edge):** Intra‑zone orchestrator; runs clustering, holds **cluster models**; aggregates UE updates.
- **Satellite (global server):** Cross‑zone knowledge sharing and consolidation; runs global/zone‑aware aggregation.
- **Cluster model (per zone):** Model parameters maintained per cluster c in zone z.
- **Global model:** Optional satellite‑level reference used for seeding/synchronizing cluster models.

---

## 4) Data & Signals for Clustering (in‑zone)
- **Model‑update similarity:** e.g., cosine similarity of Δw_i.
- **Loss/score profiles:** UE evaluates candidate cluster centroids and picks argmin loss.
- **Feature embeddings:** Pooled features/statistics (careful with privacy).
- **Metadata (optional):** Location/device hints (use only if compliant with policy).

**When to (re)cluster:**
- Initial warm‑up (few global rounds) and **periodically every K_zone rounds** or when drift detected (cluster inertia ↓, assignment churn ↑).

**Constraints:**
- Minimum cluster size m_min.
- Orphan UEs fall back to nearest cluster or temporary personal model.

---

## 5) Algorithm Menu (What to Run Where)

### 5.1 Synchronous (round‑based) HFL baselines
**(A) Hierarchical FedAvg (H‑FedAvg)**
- **Flow:**
  1) **UE local:** Each selected UE trains E steps from its assigned **cluster model** (or zone seed if unassigned).
  2) **Zone aggregation (UAV):** Aggregate UE updates **per cluster** via weighted FedAvg; update cluster models.
  3) **Global aggregation (Satellite):** Every τ_sat rounds, UAVs send **zone summaries** (e.g., averaged cluster backbones or a single zone model). Satellite aggregates into a **global backbone** using FedAvg/optimizer variants (FedAdam/FedYogi). New global backbone seeds the next zone cycle.
- **Notes:**
  - Keeps personalization at the **cluster level**; maintains a light coupling via the satellite backbone.
  - Simple, robust; good starting point.

**(B) H‑FedAvg + Prox (FedProx at UE)**
- Add proximal term μ‖w − w_ref‖² to stabilize under heterogeneity/stragglers.
- **w_ref** can be **cluster model** (preferred) or global backbone.

**(C) H‑FedAvg + Control variates (SCAFFOLD‑style)**
- Mitigates client drift with control variates at UE/UAV; higher overhead but stabilizes learning on non‑IID.

### 5.2 Synchronous PFL variants (personalization on top)
**(D) Per‑FedAvg / Fine‑tune head**
- Share **backbone**; keep **UE‑specific head** (or few last layers). After zone/global sync, each UE fine‑tunes head locally.

**(E) pFedMe / Ditto (prox‑personalization)**
- Each UE optimizes a **personal model φ_i** while participating in shared model updates.
- **pFedMe:** Local objective f_i(w) + (λ/2)‖w − φ_i‖² with alternating updates.
- **Ditto:** Train shared model and **a personal copy** with a proximal penalty, simple to implement.

**(F) APFL (mixture)**
- UE keeps a convex combination: **θ_i = α·w_shared + (1−α)·w_personal**, with α tuned per UE.

### 5.3 Synchronous Clustered FL (in‑zone)
**(G) IFCA/CFL‑style cluster assignment (zone‑local)**
- UAV maintains **C** cluster centroids/models {w_c}.
- **Assignment:** At round start, UE evaluates small steps or losses against each w_c, chooses c* = argmin loss.
- **Update:** Per cluster c, aggregate assigned UEs via FedAvg; refresh w_c.
- **Reseeding:** If empty/unstable clusters → merge/split; optionally re‑init from global backbone.

---

## 6) Asynchronous Algorithms (UE→UAV and UAV→Satellite)

### 6.1 Asynchronous UE→UAV (per‑cluster server state)
**(H) FedAsync‑style with staleness weighting**
- **Server (UAV) state per cluster c:** current model w_c^t and version t.
- **UE push:** When a UE finishes local steps E, it sends (Δw_i, t_i, |D_i|, id_cluster_hint).
- **Aggregation on arrival:**
  - Compute **staleness s = t − t_i**.
  - Weight update: **η_eff = η · γ^min(s, S_max)** (γ∈(0,1), S_max cutoff).
  - Optionally normalize by data size |D_i|/Σ|D|.
  - Update: **w_c ← w_c + η_eff · Δw_i**.
- **Admission control:** Drop or down‑weight updates older than S_max; optionally ask UE to refresh base.

**(I) FedBuff‑style buffered async**
- UAV buffers up to **B** updates (or Δt seconds) then performs an aggregate step (FedAvg/FedAdam). Reduces variance from purely one‑by‑one updates.

**(J) Async + Prox / Control variates**
- Combine (H)/(I) with **prox** (FedProx) or **control variates** (SCAFFOLD) for stability under heavy skew.

### 6.2 Asynchronous UAV→Satellite
**(K) Buffered global updates (zone summaries)**
- Each UAV maintains a timer or update counter. On trigger, send a **zone summary**: e.g., averaged cluster backbone, moments (m,v) if FedAdam used at UAV.
- Satellite aggregates asynchronously using staleness weighting (same γ,S_max idea) or FedBuff at core.

**(L) Event‑driven distillation (optional)**
- If model sizes differ or we want privacy‑amplified transfer, UAV uploads **synthetic logits/statistics**; satellite distills into a global backbone.

---

## 7) Round Clocks & Scheduling
- **Synchronous mode:**
  - **UE round window D_UE:** UEs not finished by D_UE are **dropped this round** (partial participation).
  - **UAV zone round τ_zone:** After D_UE, UAV aggregates per cluster once.
  - **Satellite global period τ_sat:** Every τ_sat zone rounds, satellite aggregates.
- **Asynchronous mode:**
  - **No hard round windows.** Use **buffers** (B) and **staleness caps** (S_max) at each tier.
  - **Priority:** Prefer fresher, larger |D_i| updates; garbage collect stale versions.

---

## 8) Model Topology & Personalization Pattern
- **Backbone + Head:** Share backbone globally/zone‑wise; personalize heads per cluster or per UE.
- **Cluster‑only personalization:** Maintain one head per cluster at UAV; UE just fine‑tunes a tiny adapter (LoRA/BN‑stats).
- **APFL mixture:** UE blends shared vs personal weights with α (tunable per UE or learned).

---

## 9) Communication & Systems Pragmatics
- **Compression:** 8‑bit quantization, sparsification (top‑k), or sketching at UE; reconstitute at UAV.
- **Security:** (Optional) secure aggregation at UE→UAV, plus TLS. DP noise on updates if needed.
- **Resilience:** Idempotent update IDs, versioning per cluster, retry with exponential back‑off.
- **State kept:**
  - UE: personal head/adapter; last base version t_i.
  - UAV: per‑cluster params, optimizer states (m,v), assignment cache; small logs for drift.
  - Satellite: global backbone (and optionally per‑zone adapters), optimizer states.

---

## 10) Minimal Pseudocode (Text‑only)

### 10.1 Synchronous H‑FedAvg with Zone Clustering
```
# At Satellite (global cycle every τ_sat):
for g in 1..G:  # global periods
  broadcast_backbone(w_global)
  for r in 1..τ_sat:  # zone rounds between global syncs
    for each UAV_z in parallel:
      if r == 1 and (init or drift): UAV_z.cluster_reseed_from(w_global)
      # --- UE local training ---
      assign each UE_i → cluster c via loss-on-centroids or similarity
      for each UE_i selected:
        w_i ← w_c  # cluster model
        w_i ← LocalTrain(f_i, w_i, E, prox=μ optional, scaffold optional)
        send Δw_i, |D_i| to UAV_z(c)
      # --- per-cluster zone aggregation ---
      for each cluster c:
        w_c ← FedAvg({Δw_i}, weights=|D_i|)
  # --- global aggregation ---
  gather zone summaries {w_z or avg_c(w_c)}
  w_global ← FedAvg or FedAdam({w_z})
```

### 10.2 Async UE→UAV (FedAsync/FedBuff hybrid per cluster)
```
# UAV per cluster c keeps: w_c, version t, buffer B_c
on UE_i update arrival (Δw_i, t_i, |D_i|):
  s ← t - t_i
  if s > S_max: discard or request refresh
  η_eff ← η * γ^min(s, S_max)
  push (Δw_i, η_eff, |D_i|) into B_c
  if |B_c| ≥ B or timer_expired:
    apply aggregate: w_c ← w_c + Σ (η_eff * weight(|D_i|) * Δw_i)
    t ← t + 1; clear B_c
```

### 10.3 Async UAV→Satellite (Buffered global)
```
# Satellite maintains global backbone w_G, opt state (m,v)
on zone summary arrival from UAV_z at version u:
  s ← t_G - u
  α_eff ← α * γ^min(s, S_max)
  w_G ← FedBuff/FedAdamStep(w_G, summary_z, α_eff)
  t_G ← t_G + 1 (on aggregate steps)
```

---

## 11) Hyperparameters (Cheat‑Sheet)
- **Local steps E:** 1–5 (mobile) or 5–20 (stable).
- **Cluster count C per zone:** 2–5 to start; enforce m_min (≥5 UEs/cluster if possible).
- **Zone rounds τ_zone:** 1–5 between satellite syncs.
- **Global period τ_sat:** 1–5 zone rounds.
- **Async staleness:** γ ∈ [0.5, 0.99], S_max ∈ [3, 10] versions.
- **Prox μ (FedProx/ Ditto/ pFedMe):** 1e‑4–1e‑2.
- **APFL α:** 0.3–0.7 (tune per UE/cluster).
- **Buffers B (FedBuff):** UE→UAV: 8–64; UAV→Sat: 4–16.

---

## 12) Metrics & Diagnostics
- **Personal accuracy:** per‑UE test; report mean ± std across UEs.
- **Fairness:** 10th–90th percentile gap, worst‑k accuracy.
- **Cluster cohesion/separation:** silhouette score on updates; assignment churn rate.
- **Comms cost:** bytes/round/tier, compression ratio.
- **Freshness:** staleness distribution; drop rate due to S_max.
- **Convergence:** loss curves per tier; drift indicators.

---

## 13) Failure & Edge Cases
- **Stragglers:** Drop in sync mode; in async, down‑weight by staleness.
- **Empty clusters:** Merge with nearest; reseed from global.
- **Concept drift:** Increase recluster frequency; raise μ; enable control variates.
- **UAV loss:** Reconstitute from satellite backbone + last saved cluster states.
- **Outliers:** Cap per‑UE norm (clip Δw), robust aggregation (median/trimmed mean).

---

## 14) Implementation Notes
- **Versioning:** (tier, zone, cluster, step) composite IDs; monotonic clocks per tier.
- **Storage:** Lightweight key‑value (e.g., cluster_id → {w, m, v, t}).
- **Interfaces:**
  - UE→UAV: `{ue_id, cluster_hint, base_ver, Δw, |D|, stats}`
  - UAV→Sat (summary): `{zone_id, ver, w_zone_or_avg, opt_moments?, counts}`
- **Privacy add‑ons:** Secure aggregation at UE tier; DP noise on Δw.
- **Compression:** Quantize at UE; dequantize at UAV.

---

## 15) Recommended Starting Config (Normalized)
1) **Sync first:** H‑FedAvg with in‑zone IFCA‑style clustering; backbone shared via satellite every τ_sat=2.
2) **Personalization:** UE keeps a small head (Per‑FedAvg) or APFL with α=0.5.
3) **Stability:** FedProx μ=1e‑3; optional SCAFFOLD if drift observed.
4) **Then enable Async:** UE→UAV FedBuff with B=16, γ=0.9, S_max=5; UAV→Sat buffer B=8.

---

## 16) To‑Do Hooks
- [ ] Define concrete feature/update similarity for clustering (cosine on Δw vs loss‑based).
- [ ] Decide C per zone (grid search or silhouette heuristics).
- [ ] Pick optimizer at UAV/Satellite (FedAdam default) and learning rates.
- [ ] Implement versioned, idempotent message bus.
- [ ] Build metrics dashboards (freshness, fairness, churn).

---

## 17) Quick Glossary
- **E:** local training steps per UE before sending an update.
- **τ_zone / τ_sat:** zone/global sync cadence in synchronous mode.
- **γ, S_max:** async staleness decay and cutoff.
- **μ:** proximal weight to limit drift.
- **APFL α:** mix shared vs personal model proportion.
- **FedBuff:** buffered async aggregation; **FedAsync:** immediate async with staleness weights.
- **IFCA/CFL:** clustered FL procedures assigning clients to per‑cluster models.

