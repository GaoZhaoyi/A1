#!/usr/bin/env python3
import argparse
import logging
import os
import random
import numpy as np

import torch
import torch.nn as nn

# Import our custom modules
from data_download import load_cifar_dataset
from model_architectures import create_model
from train_utils import (
    create_data_loaders,
    define_loss_and_optimizer,
    train_epoch,
    validate_epoch,
    evaluate_model,
    save_checkpoint
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--batch_size", type=int, default=256)  # Larger batch size for faster training
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)  # Higher learning rate
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/models", exist_ok=True)

    # Load datasets (NO augmentation at this stage to prevent data leakage)
    print("📥 Loading datasets...")
    train_dataset, _ = load_cifar_dataset(args.dataset, args.data_dir, train=True)
    test_dataset, _ = load_cifar_dataset(args.dataset, args.data_dir, train=False)

    # Determine number of classes
    num_classes = 10 if args.dataset == "cifar10" else 100

    # Create data loaders (this ensures no data leakage)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, args.batch_size
    )

    # Create model
    print("🧠 Creating model...")
    model = create_model(num_classes, args.device)

    # Define loss, optimizer, and scheduler
    criterion, optimizer, scheduler = define_loss_and_optimizer(
        model, args.lr, args.weight_decay
    )

    # Update scheduler steps per epoch
    scheduler.total_steps = len(train_loader) * args.num_epochs

    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 10

    print(f"🔥 Starting training for {args.num_epochs} epochs...")

    for epoch in range(args.num_epochs):
        # Update scheduler steps per epoch for OneCycleLR
        if epoch == 0:
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.lr,
                epochs=args.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )

        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, args.device
        )

        # Validate epoch
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, args.device)

        # Print epoch results
        print(f"Epoch {epoch + 1:2d}/{args.num_epochs}: "
              f"Train Acc: {train_acc:6.2f}%, Val Acc: {val_acc:6.2f}%, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
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

        # Test evaluation every 10 epochs using best model
        if (epoch + 1) % 10 == 0:
            # Load best model for testing
            checkpoint = torch.load(f"{args.output_dir}/models/best_model.pth")
            model.load_state_dict(checkpoint["state_dict"])

            # Evaluate on test set
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, args.device)
            print(f"  🧪 Test Accuracy at epoch {epoch + 1}: {test_acc:.2f}%")

            # Save test results
            with open(f"{args.output_dir}/test_results_epoch_{epoch + 1}.txt", "w") as f:
                f.write(f"Epoch {epoch + 1} Test Results:\n")
                f.write(f"Test Accuracy: {test_acc:.2f}%\n")
                f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")

            # Restore current model state
            model.load_state_dict(checkpoint["state_dict"])

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"⚠️ Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation with best model
    print("\n🏁 Final evaluation with best model...")
    checkpoint = torch.load(f"{args.output_dir}/models/best_model.pth")
    model.load_state_dict(checkpoint["state_dict"])

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, args.device)
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")
    print(f"📈 Best Validation Accuracy: {best_val_acc:.2f}%")

    # Save final results
    with open(f"{args.output_dir}/final_results.txt", "w") as f:
        f.write(f"Final Results:\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Training completed in {checkpoint['epoch']} epochs\n")

    print(f"✅ Training completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
