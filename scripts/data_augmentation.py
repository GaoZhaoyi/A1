import logging
from pathlib import Path
from typing import Optional, Tuple

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cifar_transforms(train=True):
    """Get appropriate transforms for training or testing"""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def load_cifar_dataset(dataset_name="cifar100", root_dir="data", train=True):
    """Load CIFAR dataset directly without saving to folders"""
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    transform = get_cifar_transforms(train)

    if dataset_name == "cifar10":
        dataset = CIFAR10(root=str(root_dir), train=train, download=True, transform=transform)
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    else:  # cifar100
        dataset = CIFAR100(root=str(root_dir), train=train, download=True, transform=transform)
        class_names = None  # CIFAR-100 has too many classes to list

    logger.info(f"Loaded {dataset_name} {'train' if train else 'test'} dataset: {len(dataset)} samples")
    return dataset, class_names
