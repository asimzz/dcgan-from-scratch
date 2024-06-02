from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def load_dataset(dataset_path: str, transform: transforms.Compose, batch_size: str):
    dataset = ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
