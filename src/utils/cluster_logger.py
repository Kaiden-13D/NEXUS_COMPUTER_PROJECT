import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

def log_cluster_assignments(
    experiment_name: str,
    round_num: int,
    selected_clients: List, # List of FLClient objects
    client_to_cluster: Dict[str, int]
):
    """
    Logs the cluster assignment for each selected client to a CSV file.

    Args:
        experiment_name: The name of the experiment (e.g., 'bl3', 'abl1').
        round_num: The current communication round.
        selected_clients: A list of the FLClient objects that were selected for training.
        client_to_cluster: A dictionary mapping client_id to cluster_id.
    """
    results_dir = Path(f"results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "cluster_assignments.csv"

    # Write header if file doesn't exist
    if not log_file.exists():
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["round", "client_id", "cluster_id"])

    # Append data for the current round
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for client in selected_clients:
            cluster_id = client_to_cluster.get(client.id, -1) # Default to -1 if not found
            writer.writerow([round_num, client.id, cluster_id])
