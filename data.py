
from collections import defaultdict

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



def partition_data(dataset, num_clients, scenario='non-iid-label'):
    """
    Partitions the dataset for a number of clients more efficiently.
    This version avoids multiple .filter() calls.
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
            client_indices.extend(writer_to_indices[writer_id]) # extend 리스트 뒤에 붙이기.
        
        # 4. Create a Subset of the original HF dataset
        client_subset = dataset.select(client_indices)
        
        transform = Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
        """1. The Transformation Pipeline (transform)
        
        transform = Compose([...]) 
        creates a single pipeline that chains several transformation steps together.
         When an image is passed to transform, it goes through these steps in order:

         ToTensor(): This is the first step.
         It converts the input image (which is likely a PIL Image or NumPy array) into a PyTorch Tensor.
         It scales the image's pixel values. Image pixels are typically in the range [0, 255]. 
         ToTensor() converts them into a floating-point tensor with values in the range [0.0, 1.0].
         It changes the tensor's dimension order from $H \times W \times C$ (Height, Width, Channel) 
         to $C \times H \times W$ (Channel, Height, Width), which is the format PyTorch models expect
         
         .Normalize((0.5,), (0.5,)): This is the second step, applied after ToTensor.
         It normalizes the tensor's values using a given mean and standard deviation. 
         The formula is: $output = (input - mean) / std$.
         
         In your code, the $mean$ is $0.5$ and the $std$ (standard deviation) is $0.5$.
         This step effectively shifts the [0.0, 1.0] range to [-1.0, 1.0].
         
         Min value: $(0.0 - 0.5) / 0.5 = -1.0
         $Max value: $(1.0 - 0.5) / 0.5 = 1.0
         
         $Why do this? Normalizing input data to be centered around 0 
         (like in the [-1, 1] range) helps the neural network train more efficiently and stably. 

        The (0.5,) tuple format implies the images are single-channel (grayscale), which is correct for the FEMNIST dataset."""
        
        client_datasets.append(FEMNISTDataset(client_subset, transform=transform))
        
    return client_datasets

if __name__ == '__main__':
    # Example usage
    full_dataset = load_femnist_dataset()
    print(f"Total samples: {len(full_dataset)}")
    print(f"Features: {full_dataset.features}")

    num_clients = 120
    client_datasets = partition_data(full_dataset, num_clients)

    print(f"\nPartitioned data for {num_clients} clients.")
    for i, client_ds in enumerate(client_datasets):
        print(f"Client {i+1} has {len(client_ds)} samples.")

