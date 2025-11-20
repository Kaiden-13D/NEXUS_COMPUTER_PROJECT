"""
FEMNIST dataset loading and writer-based partitioning
"""
import random
from collections import defaultdict
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

from ..utils.progress_logger import get_progress_logger
from ..config.default_config import Config


class FEMNISTDataset(Dataset):
    """FEMNIST dataset wrapper"""
    
    def __init__(self, dataset, transform=None, noise_ratio=0.0, num_classes=62):
        self.dataset = dataset
        self.transform = transform
        self.noise_ratio = noise_ratio
        self.num_classes = num_classes
        
        # Pre-compute noise indices if noise is enabled
        self.noise_indices = set()
        if self.noise_ratio > 0:
            total_samples = len(self.dataset)
            num_noisy = int(total_samples * self.noise_ratio)
            self.noise_indices = set(random.sample(range(total_samples), num_noisy))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['character']
        
        # Apply label noise
        if idx in self.noise_indices:
            # Pick a random label different from the true label
            original_label = label
            while label == original_label:
                label = random.randint(0, self.num_classes - 1)
        
        # Convert RGB to grayscale if needed
        if hasattr(image, 'convert') and image.mode != 'L':
            image = image.convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


def setup_femnist_by_writer(
    num_clients: int, 
    test_split_ratio: float = 0.1, 
    max_samples: Optional[int] = None
) -> Tuple[List[FEMNISTDataset], List[FEMNISTDataset], int]:
    """
    Distribute FEMNIST data to clients based on writer ID
    
    Args:
        num_clients: Number of clients
        test_split_ratio: Test set ratio
        max_samples: Maximum samples per client (None for unlimited)
    
    Returns:
        (client_train_datasets, client_test_datasets, num_classes)
    """
    print("Loading FEMNIST from Hugging Face (flwrlabs/femnist)...")
    hf_dataset = load_dataset("flwrlabs/femnist", split="train")
    
    print("Grouping data by writer_id... (this may take a while)")
    writer_to_indices = defaultdict(list)
    
    logger = get_progress_logger()
    if logger:
        logger.log("Grouping data by writer_id...", print_to_console=False)
    
    # Show progress with tqdm (logged to progress log only)
    try:
        from tqdm import tqdm
        # Redirect tqdm output to file
        import sys
        original_stdout = sys.stdout
        if logger:
            # Redirect tqdm output to progress log file
            tqdm_file = open(logger.log_file, 'a') if logger else None
            writer_ids_iter = tqdm(enumerate(hf_dataset['writer_id']), 
                                   total=len(hf_dataset),
                                   desc="Grouping by writer_id",
                                   file=tqdm_file if tqdm_file else sys.stdout)
        else:
            writer_ids_iter = tqdm(enumerate(hf_dataset['writer_id']), 
                                   total=len(hf_dataset),
                                   desc="Grouping by writer_id")
    except ImportError:
        writer_ids_iter = enumerate(hf_dataset['writer_id'])
        tqdm_file = None
    
    try:
        for i, writer_id in writer_ids_iter:
            writer_to_indices[writer_id].append(i)
    finally:
        if 'tqdm_file' in locals() and tqdm_file:
            tqdm_file.close()
            sys.stdout = original_stdout
    
    writer_ids = sorted(list(writer_to_indices.keys()))
    print(f"Total writers: {len(writer_ids)}")
    
    # Determine distribution mode
    dist_mode = getattr(Config, "DATA_DISTRIBUTION", "tiered")
    print(f"Data distribution mode: {dist_mode}")
    
    # Common transform
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    client_train_datasets = []
    client_test_datasets = []
    
    logger = get_progress_logger()
    if logger:
        logger.log(f"Creating datasets for {num_clients} clients...", print_to_console=False)
    
    # Show progress for creating client datasets (logged to progress log only)
    try:
        from tqdm import tqdm
        import sys
        if logger:
            tqdm_file = open(logger.log_file, 'a')
            clients_iter = tqdm(range(num_clients), desc="Creating client datasets", file=tqdm_file)
        else:
            clients_iter = tqdm(range(num_clients), desc="Creating client datasets")
    except ImportError:
        clients_iter = range(num_clients)
        tqdm_file = None
    
    try:
        # Tiered distribution settings
        if dist_mode == "tiered":
            # Good: 20%, Medium: 50%, Bad: 30%
            num_good = int(num_clients * 0.2)
            num_medium = int(num_clients * 0.5)
            num_bad = num_clients - num_good - num_medium
            
            # Calculate writers per client type
            # Total writers W. Let avg = W / N.
            # Good: ~2 * avg, Bad: ~0.5 * avg, Medium: ~avg
            # Constraint: num_good * w_good + num_medium * w_medium + num_bad * w_bad = W
            
            avg_writers = len(writer_ids) / num_clients
            w_good = int(avg_writers * 2.0)
            w_bad = max(1, int(avg_writers * 0.5))
            
            # Adjust medium to fit remaining
            remaining_writers = len(writer_ids) - (num_good * w_good) - (num_bad * w_bad)
            w_medium = max(1, remaining_writers // num_medium) if num_medium > 0 else 0
            
            # Assign tiers
            tiers = (["good"] * num_good) + (["medium"] * num_medium) + (["bad"] * num_bad)
            random.shuffle(tiers)
            
            current_writer_idx = 0
        else:
            writers_per_client = len(writer_ids) // num_clients
            tiers = ["default"] * num_clients
            current_writer_idx = 0

        for i in clients_iter:
            tier = tiers[i]
            
            if dist_mode == "tiered":
                if tier == "good":
                    n_writers = w_good
                elif tier == "bad":
                    n_writers = w_bad
                else:
                    n_writers = w_medium
            else:
                n_writers = writers_per_client
            
            # Assign writers
            end_idx = min(current_writer_idx + n_writers, len(writer_ids))
            client_writers = writer_ids[current_writer_idx:end_idx]
            current_writer_idx = end_idx
            
            # Collect data indices from assigned writers
            client_indices = []
            for wid in client_writers:
                client_indices.extend(writer_to_indices[wid])
            
            # (Optional) Limit maximum samples per client
            if max_samples and len(client_indices) > max_samples:
                random.shuffle(client_indices)
                client_indices = client_indices[:max_samples]
            
            # Train/Test split
            random.shuffle(client_indices)
            split_pt = int(len(client_indices) * (1 - test_split_ratio))
            train_idx = client_indices[:split_pt]
            test_idx = client_indices[split_pt:]
            
            # HF Dataset's .select() creates subset while preserving metadata
            train_sub = hf_dataset.select(train_idx)
            test_sub = hf_dataset.select(test_idx)
            
            # Apply Label Noise for "Bad" clients
            if dist_mode == "tiered" and tier == "bad":
                noise_ratio = getattr(Config, "LABEL_NOISE_RATIO", 0.1)
                # We need to wrap the dataset to apply noise on-the-fly or modify it
                # Here we use a wrapper that flips labels
                client_train_datasets.append(FEMNISTDataset(train_sub, transform=transform, noise_ratio=noise_ratio, num_classes=62))
            else:
                client_train_datasets.append(FEMNISTDataset(train_sub, transform=transform))
            
            client_test_datasets.append(FEMNISTDataset(test_sub, transform=transform))
            
    finally:
        if 'tqdm_file' in locals() and tqdm_file:
            tqdm_file.close()
            if logger:
                import sys
                sys.stdout = sys.__stdout__
    
    print(f"Done. Created {len(client_train_datasets)} client datasets.")
    return client_train_datasets, client_test_datasets, 62  # 62 classes

