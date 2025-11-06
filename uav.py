"""
UAV representation for the federated learning simulator.

This module defines the UAV class, which acts as an aggregator for a group of clients.
UAVs are responsible for dynamic client selection (DCS) and aggregating updates from clients.
"""

from collections import OrderedDict
from dcs import compute_scores

class UAV:
    """Represents a UAV that aggregates updates from a group of clients."""
    def __init__(self, uav_id, clients):
        self.uav_id = uav_id
        self.clients = clients
        self.shared_model_state = None # This will be updated by the satellite

    def select_clients(self, m, weights):
        """Selects the top m clients based on their DCS scores.

        Args:
            m (int): The number of clients to select.
            weights (dict): The weights for the DCS scoring formula.

        Returns:
            A list of the selected Client objects.
        """
        if not self.clients:
            return []
            
        client_scores = compute_scores(self.clients, weights)
        
        # Sort clients by score in descending order
        sorted_clients = sorted(client_scores, key=lambda x: x[1], reverse=True)
        
        # Select the top m clients
        selected_clients = [client for client, score in sorted_clients[:m]]
        
        return selected_clients

    def aggregate_updates(self, client_updates):
        """Aggregates parameter updates from selected clients.

        This implementation uses Federated Averaging (FedAvg), where updates are
        weighted by the number of data samples each client used for training.

        Args:
            client_updates (list): A list of tuples, where each tuple contains
                                   (deltas, num_samples).

        Returns:
            An OrderedDict with the aggregated model parameter deltas.
        """
        if not client_updates:
            return OrderedDict()

        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples == 0:
            return OrderedDict()

        # Initialize aggregated deltas with zeros
        aggregated_deltas = OrderedDict()
        for key in client_updates[0][0].keys():
            aggregated_deltas[key] = 0.0

        # Perform weighted averaging
        for deltas, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in deltas:
                aggregated_deltas[key] += deltas[key] * weight
        
        return aggregated_deltas
