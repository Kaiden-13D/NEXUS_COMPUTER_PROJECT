# Agent Blueprint 2: Architectural Patterns in Hierarchical FL

## 0) Purpose & Scope
This document summarizes two fundamental architectural patterns for handling user diversity (non-IID data) in a hierarchical federated learning system. It clarifies the distinction between using clustering as an **intermediate tool** versus using it as a **final objective**.

---

## 1) Pattern 1: Single Global Model with Local Clustering

This is the standard architecture we are implementing in this project.

- **Concept:** The system's ultimate goal is to produce **one single, highly-generalized global model**. To achieve this more effectively, clustering is used as an intermediate optimization step that occurs **locally within each zone**.

- **Mechanism:**
    1.  **Local Clustering:** The UAV (zone server) groups the clients *within its own zone* into temporary clusters based on their current data similarity (e.g., via loss-based assignment).
    2.  **Intra-Cluster Aggregation:** Updates from clients are first averaged within their compatible cluster, reducing the "destructive interference" caused by conflicting data. This produces higher-quality, more stable cluster models.
    3.  **Zone Summary:** The UAV averages its local cluster models into a single "zone summary" model.
    4.  **Global Aggregation:** The satellite averages the zone summaries from all UAVs to update the single global model.

- **Motivation:** The primary motivation is **stability and efficiency**. This pattern is not designed to produce different models for different users. Instead, it uses local clustering as a clever tool to manage the chaos of non-IID data, ensuring that the updates being sent up the hierarchy are of higher quality. This leads to faster, more reliable convergence for the **single global model**.

- **Best For:** **Single-task environments** where the objective is one robust model that works well for everyone (e.g., a general image classifier like our MNIST project, a foundational language model).

- **Analogy:** An **organized workshop**. A manager (UAV) organizes employees into small, focused breakout groups (local clusters) to refine their ideas first. The manager then combines these high-quality, refined ideas into a single, coherent final report (the global model).

---

## 2) Pattern 2: Multiple Global Models with Global Clustering

This is the alternative architecture we discussed, suitable for different problem types.

- **Concept:** The system's goal is to produce **multiple, distinct, specialized global models**, each tailored to a persistent, well-defined category of user or task that exists across the entire network.

- **Mechanism:**
    1.  **Global Categories:** The system defines several "global clusters" or categories (e.g., "Task A" vs. "Task B", or "Professional Users" vs. "Casual Users").
    2.  **Global Assignment:** The satellite assigns each client or zone to one of these global categories.
    3.  **Partitioned Aggregation:** The satellite maintains a separate global model for each category. When it receives a zone summary, it only uses it to update the specific global model corresponding to that zone's category.

- **Motivation:** The motivation is **explicit, large-scale personalization**. This pattern is used when a "one-size-fits-all" model is known to be insufficient and the goal is to provide distinct, optimized experiences for different user segments.

- **Best For:** **Multi-task environments** or scenarios with fundamentally different types of users or data that are known in advance.

- **Analogy:** A **large corporation with specialized departments**. The headquarters (satellite) doesn't try to create one "master employee" model. It trains a specialized model for the Engineering department and a completely different one for the Marketing department, because their jobs and data are fundamentally different.

- **Example Scenario:** A healthcare network with **Large Research Hospitals** and **Small Regional Clinics**. A single model would fail for both. The correct approach is to train two separate global models: a `Global_Model_Hospital` and a `Global_Model_Clinic`.

---

## 3) Summary of Differences

| Feature | Pattern 1: Single Global Model (Our Project) | Pattern 2: Multiple Global Models |
| :--- | :--- | :--- |
| **Primary Goal** | One robust, generalized model | Multiple specialized models |
| **Number of Global Models** | **One** | **Multiple** |
| **Scope of Clustering** | **Local** (within each zone) | **Global** (across the whole network) |
| **Role of Clustering** | An **intermediate tool** for stability | The **final objective** of personalization |
| **Best Use Case** | Single, unified task | Multiple tasks or distinct user categories |

**Conclusion:** The choice of architecture is driven by the problem definition. For our project's goal of building the best possible MNIST classifier for all users, **Pattern 1 is the correct and standard design.**
