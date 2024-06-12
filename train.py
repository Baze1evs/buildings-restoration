import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def train(model, source, target, epochs, batch_size):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(256, 256), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    source_domain = ImageFolder(
        root=source,
        transform=transforms,
    )

    target_domain = ImageFolder(
        root=target,
        transform=transforms,
    )

    damaged_dataloader = DataLoader(source_domain, batch_size=batch_size, shuffle=True)
    pristine_dataloader = DataLoader(target_domain, batch_size=batch_size, shuffle=True)

    model.train(epochs, damaged_dataloader, pristine_dataloader)
