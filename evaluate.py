import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def evaluate(model, source, target):
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_data = ImageFolder(
        root=source,
        transform=test_transform,
    )

    test_dataloader = DataLoader(test_data, batch_size=20, shuffle=False, drop_last=False)

    for i, batch in enumerate(test_dataloader):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        batch = batch.to(model.device)
        with torch.no_grad():
            pred = model.gen(batch)

        for j, arr in enumerate(pred):
            v2.functional.to_pil_image(arr * 0.5 + 0.5).save(f"{target}/{i*len(batch)+j}.jpg")
