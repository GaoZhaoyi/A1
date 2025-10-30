#!/usr/bin/env python3
import argparse
import logging
import os
import random
import numpy as np

import torch
import torch.nn as nn
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

# 使用相对导入或直接复制必要函数
def load_cifar_dataset(dataset_name="cifar100", root_dir="data", train=True):
    """Load CIFAR dataset with proper normalization"""

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR-100官方统计值
    if dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:  # CIFAR-10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if dataset_name == "cifar10":
        dataset = CIFAR10(root=str(root_dir), train=train, download=True, transform=transform)
    else:  # cifar100
        dataset = CIFAR100(root=str(root_dir), train=train, download=True, transform=transform)

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded {dataset_name} {'train' if train else 'test'} dataset: {len(dataset)} samples")
    return dataset


def create_data_loaders(train_dataset, test_dataset, batch_size, validation_split=0.2, seed=42):
    """Create train/val/test data loaders without data leakage"""
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    np.random.seed(seed)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    split_idx = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Data split - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


class OptimizedCNN(nn.Module):
    """Optimized CNN for fast convergence on CIFAR-100"""

    def __init__(self, num_classes=100):
        super(OptimizedCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.dropout(x)

        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def create_model(num_classes, device):
    """Create and initialize the model"""
    model = OptimizedCNN(num_classes=num_classes)
    model = model.to(device)
    return model


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Define optimized loss function and optimizer"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-4)
    return criterion, optimizer


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch with optimized settings"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    from tqdm import tqdm
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    from tqdm import tqdm
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)


def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimized CIFAR-100 Training Pipeline")

    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    set_random_seeds(args.seed)

    print("🚀 Starting optimized CIFAR-100 training...")
    print(f"Using device: {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/models", exist_ok=True)

    print("📥 Loading datasets...")
    train_dataset = load_cifar_dataset(args.dataset, args.data_dir, train=True)
    test_dataset = load_cifar_dataset(args.dataset, args.data_dir, train=False)

    num_classes = 10 if args.dataset == "cifar10" else 100

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, args.batch_size
    )

    print("🧠 Creating model...")
    model = create_model(num_classes, args.device)

    criterion, optimizer = define_loss_and_optimizer(
        model, args.lr, args.weight_decay
    )

    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 10

    print(f"🔥 Starting training for {args.num_epochs} epochs...")

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        val_loss, val_acc = validate_epoch(model, val_loader, criterion, args.device)

        print(f"Epoch {epoch + 1:2d}/{args.num_epochs}: "
              f"Train Acc: {train_acc:6.2f}%, Val Acc: {val_acc:6.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "optimizer": optimizer.state_dict(),
            }, f"{args.output_dir}/models/best_model.pth")
            print(f"  🎯 New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            checkpoint = torch.load(f"{args.output_dir}/models/best_model.pth")
            model.load_state_dict(checkpoint["state_dict"])

            test_loss, test_acc = evaluate_model(model, test_loader, criterion, args.device)
            print(f"  🧪 Test Accuracy at epoch {epoch + 1}: {test_acc:.2f}%")

            with open(f"{args.output_dir}/test_results_epoch_{epoch + 1}.txt", "w") as f:
                f.write(f"Epoch {epoch + 1} Test Results:\n")
                f.write(f"Test Accuracy: {test_acc:.2f}%\n")
                f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")

            model.load_state_dict(checkpoint["state_dict"])

        if patience_counter >= patience_limit:
            print(f"⚠️ Early stopping at epoch {epoch + 1}")
            break

    print("\n🏁 Final evaluation with best model...")
    checkpoint = torch.load(f"{args.output_dir}/models/best_model.pth")
    model.load_state_dict(checkpoint["state_dict"])

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, args.device)
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")
    print(f"📈 Best Validation Accuracy: {best_val_acc:.2f}%")

    with open(f"{args.output_dir}/final_results.txt", "w") as f:
        f.write(f"Final Results:\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Training completed in {checkpoint['epoch']} epochs\n")

    print(f"✅ Training completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
