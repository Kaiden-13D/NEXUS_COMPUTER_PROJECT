


from collections import defaultdict
import random

from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
import torch

class FEMNISTDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['character']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def load_femnist_dataset(split='train'):
    """
    Loads the FEMNIST dataset from Hugging Face.
    """
    return load_dataset("flwrlabs/femnist", split=split)

def partition_data(dataset, num_clients, scenario='non-iid-label', max_samples_per_client=None, test_split_ratio=0.1):
    """
    Partitions the dataset for a number of clients, creating train/test splits for each.
    """
    # 1. Group indices by writer_id in a single pass
    print("   - Grouping data by writer...")
    writer_to_indices = defaultdict(list) # 이거 그냥 key 없으면 자동으로 에러 안내고 key 만들어주는 딕셔너리임. 
    # This assumes the dataset is a Hugging Face Dataset object
    for i, writer_id in enumerate(dataset['writer_id']):
        writer_to_indices[writer_id].append(i)
    
    writer_ids = sorted(list(writer_to_indices.keys()))
    print("writer_ids sample: ", writer_ids[:10])
    
    # 2. Distribute writer_ids to clients
    writers_per_client = len(writer_ids) // num_clients
    print("총 writer 수: ", len(writer_ids))
    print("writer per client: ", writers_per_client)
    client_datasets = []
    
    print(f"   - Assigning writers to {num_clients} clients...")
    for i in range(num_clients):
        client_indices = []
        start_writer_idx = i * writers_per_client
        end_writer_idx = (i + 1) * writers_per_client if i < num_clients - 1 else len(writer_ids)
        
        client_writer_ids = writer_ids[start_writer_idx:end_writer_idx]
        
        # 3. Gather indices for the client's assigned writers
        for writer_id in client_writer_ids:
            client_indices.extend(writer_to_indices[writer_id])
        
        # random.shuffle(client_indices) should not shuflle to make non iid

        if max_samples_per_client and len(client_indices) > max_samples_per_client:
            client_indices = client_indices[:max_samples_per_client]

        # Split indices into training and testing
        split_idx = int(len(client_indices) * (1 - test_split_ratio))
        train_indices = client_indices[:split_idx]
        test_indices = client_indices[split_idx:]

        # Create subsets for train and test
        train_subset = dataset.select(train_indices)
        test_subset = dataset.select(test_indices)
        
        transform = Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
        
        train_dset = FEMNISTDataset(train_subset, transform=transform)
        test_dset = FEMNISTDataset(test_subset, transform=transform)
        
        client_datasets.append((train_dset, test_dset))
        
    return client_datasets

if __name__ == '__main__':
    full_dataset = load_femnist_dataset()
    print(f"Total samples: {len(full_dataset)}")

    num_clients = 10
    client_datasets = partition_data(full_dataset, num_clients, max_samples_per_client=200)

    print(f"\nPartitioned data for {num_clients} clients.")
    for i, (train_ds, test_ds) in enumerate(client_datasets):
        print(f"Client {i+1}: Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
