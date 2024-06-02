import torch


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cuda = device == "cuda"

    return (device, cuda)
