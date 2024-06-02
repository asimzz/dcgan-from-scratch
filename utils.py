import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image


def load_dataset(
    transform: transforms.Compose, batch_size: str, dataset_path: str = "dataset"
):
    dataset = ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cuda = device == "cuda"

    return (device, cuda)


def save_batch_images(batches_done, generated_image, results_path="results"):
    os.makedirs(f"{results_path}/images", exist_ok=True)

    if batches_done % 30 == 0:
        save_image(
            generated_image.data[:25],
            f"{results_path}/images/{batches_done}.png",
            nrow=5,
            normalize=True,
        )
