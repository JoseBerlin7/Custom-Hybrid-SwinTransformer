import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

def get_dataloaders(data_dir, batch_size=64, distributed=False, num_workers=8, world_size=1, rank=0):
    # train_dir = os.path.join(data_dir, "ILSVRC2012_img_train")
    train_dir = os.path.join(data_dir, "ILSVRC2012_img_trainset")
    val_dir = os.path.join(data_dir, "ILSVRC2012_img_val")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, val_tf) if os.path.exists(val_dir) else None

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if val_ds else None
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )

    return train_loader, val_loader, train_sampler, val_sampler
