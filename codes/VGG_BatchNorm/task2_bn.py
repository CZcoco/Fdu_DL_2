"""
PJ2 Task 2: Batch Normalization Analysis
- VGG-A vs VGG-A+BN performance comparison
- Loss landscape (min/max band across learning rates)
- Gradient predictiveness
- Beta-smoothness (max gradient difference)
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


PROJECT_ROOT = Path.cwd()
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def set_random_seeds(seed_value=2020, device="cpu"):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_loaders(args):
    train_loader = get_cifar_loader(
        root=str(args.data_root), batch_size=args.batch_size,
        train=True, shuffle=True, num_workers=args.num_workers,
        n_items=args.n_items, augment=True,
    )
    val_loader = get_cifar_loader(
        root=str(args.data_root), batch_size=args.batch_size,
        train=False, shuffle=False, num_workers=args.num_workers,
        n_items=args.val_items, augment=False,
    )
    return train_loader, val_loader


def train_and_record(model, optimizer, criterion, train_loader, device, epochs, scheduler=None):
    """Train and record per-step batch losses and gradient norms."""
    model.to(device)
    model.train()

    batch_losses = []
    grad_norms = []

    for epoch in tqdm(range(1, epochs + 1), unit="epoch"):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)

        if scheduler is not None:
            scheduler.step()

    return batch_losses, grad_norms


def train_with_landscape_metrics(model, optimizer, criterion, train_loader, device, epochs, sample_every=10):
    """
    Train and periodically measure:
    - gradient predictiveness: ||grad(x) - grad(x + eta*grad(x))||
    - beta-smoothness: ||grad(x + eta*d) - grad(x)|| / eta
    """
    model.to(device)
    model.train()

    batch_losses = []
    grad_predictiveness = []
    beta_smoothness = []
    step = 0

    for epoch in tqdm(range(1, epochs + 1), unit="epoch"):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            if step % sample_every == 0:
                gp, bs = _measure_landscape(model, criterion, x, y, device)
                grad_predictiveness.append(gp)
                beta_smoothness.append(bs)

            step += 1

    return batch_losses, grad_predictiveness, beta_smoothness


def _measure_landscape(model, criterion, x, y, device, eta=0.01):
    """Measure gradient predictiveness and beta-smoothness at current point."""
    model.eval()

    # Get gradient at current point
    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    grad_current = []
    for p in model.parameters():
        if p.grad is not None:
            grad_current.append(p.grad.data.clone())
        else:
            grad_current.append(torch.zeros_like(p.data))

    # Save parameters
    params_backup = [p.data.clone() for p in model.parameters()]

    # Move in gradient direction: x + eta * grad
    with torch.no_grad():
        for p, g in zip(model.parameters(), grad_current):
            p.data.add_(g, alpha=eta)

    # Get gradient at new point
    model.zero_grad()
    loss_new = criterion(model(x), y)
    loss_new.backward()
    grad_new = []
    for p in model.parameters():
        if p.grad is not None:
            grad_new.append(p.grad.data.clone())
        else:
            grad_new.append(torch.zeros_like(p.data))

    # Gradient predictiveness: ||grad_new - grad_current||
    diff_norm = 0.0
    for g1, g2 in zip(grad_current, grad_new):
        diff_norm += (g1 - g2).norm(2).item() ** 2
    grad_pred = diff_norm ** 0.5

    # Beta-smoothness: ||grad_new - grad_current|| / eta
    beta = grad_pred / eta

    # Restore parameters
    with torch.no_grad():
        for p, backup in zip(model.parameters(), params_backup):
            p.data.copy_(backup)

    model.train()
    return grad_pred, beta


def run_landscape_experiment(args):
    """Run loss landscape experiment: train with multiple LRs, plot min/max band."""
    print("\n=== Loss Landscape Experiment ===")
    device = get_device()

    # Different LR ranges for BN vs non-BN
    bn_lrs = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    no_bn_lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    landscape_data = {}

    for model_name, model_cls, lrs in [
        ("VGG-A", VGG_A, no_bn_lrs),
        ("VGG-A + BN", VGG_A_BatchNorm, bn_lrs),
    ]:
        print(f"\n  {model_name} with LRs: {lrs}")
        all_losses = []
        for lr in lrs:
            print(f"    lr={lr}...")
            set_random_seeds(args.seed, str(device))
            train_loader, _ = make_loaders(args)
            model = model_cls().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()

            losses, _ = train_and_record(model, optimizer, criterion, train_loader, device, args.epochs)
            all_losses.append(losses)

        landscape_data[model_name] = all_losses

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"VGG-A": "#3066be", "VGG-A + BN": "#d1495b"}

    for model_name, all_losses in landscape_data.items():
        min_len = min(len(l) for l in all_losses)
        aligned = np.array([l[:min_len] for l in all_losses])
        min_curve = aligned.min(axis=0)
        max_curve = aligned.max(axis=0)
        steps = np.arange(min_len)

        color = colors[model_name]
        ax.plot(steps, min_curve, color=color, linewidth=1.2, label=f"{model_name} min")
        ax.plot(steps, max_curve, color=color, linewidth=1.2, linestyle="--", label=f"{model_name} max")
        ax.fill_between(steps, min_curve, max_curve, color=color, alpha=0.15)

    ax.set_title("Loss Landscape: VGG-A vs VGG-A + BN")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Batch loss")
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "task2_loss_landscape.png", dpi=180)
    plt.close(fig)
    print(f"  Saved loss landscape plot")

    return landscape_data


def run_gradient_analysis(args):
    """Measure gradient predictiveness and beta-smoothness."""
    print("\n=== Gradient Predictiveness & Beta-Smoothness ===")
    device = get_device()

    results = {}
    for model_name, model_cls, lr in [
        ("VGG-A", VGG_A, 1e-4),
        ("VGG-A + BN", VGG_A_BatchNorm, 1e-3),
    ]:
        print(f"  Training {model_name} (lr={lr}) with landscape metrics...")
        set_random_seeds(args.seed, str(device))
        train_loader, _ = make_loaders(args)
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        losses, grad_pred, beta_smooth = train_with_landscape_metrics(
            model, optimizer, criterion, train_loader, device, args.epochs, sample_every=20
        )
        results[model_name] = {
            "losses": losses,
            "grad_predictiveness": grad_pred,
            "beta_smoothness": beta_smooth,
        }

    # Plot gradient predictiveness
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"VGG-A": "#3066be", "VGG-A + BN": "#d1495b"}

    for model_name, data in results.items():
        color = colors[model_name]
        gp = data["grad_predictiveness"]
        bs = data["beta_smoothness"]
        steps = np.arange(len(gp)) * 20

        axes[0].plot(steps, gp, color=color, linewidth=0.8, label=model_name, alpha=0.8)
        axes[1].plot(steps, bs, color=color, linewidth=0.8, label=model_name, alpha=0.8)

    axes[0].set_title("Gradient Predictiveness\n(||∇L(x+η∇L) - ∇L(x)||)")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Gradient difference norm")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].set_title("β-Smoothness\n(||∇L(x+ηd) - ∇L(x)|| / η)")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Effective β")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(args.figures_dir / "task2_gradient_analysis.png", dpi=180)
    plt.close(fig)
    print(f"  Saved gradient analysis plot")

    return {k: {"avg_grad_pred": np.mean(v["grad_predictiveness"]),
                "avg_beta": np.mean(v["beta_smoothness"])}
            for k, v in results.items()}


def run_performance_comparison(args):
    """Direct VGG-A vs VGG-A+BN training comparison."""
    print("\n=== Performance Comparison: VGG-A vs VGG-A+BN ===")
    device = get_device()

    histories = {}
    for model_name, model_cls, lr in [
        ("VGG-A (lr=1e-4)", VGG_A, 1e-4),
        ("VGG-A + BN (lr=1e-3)", VGG_A_BatchNorm, 1e-3),
    ]:
        print(f"  Training {model_name}...")
        set_random_seeds(args.seed, str(device))
        train_loader, val_loader = make_loaders(args)
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        epoch_data = {"epoch": [], "train_loss": [], "val_loss": [],
                      "train_accuracy": [], "val_accuracy": []}
        best_acc = 0

        for epoch in tqdm(range(1, args.epochs + 1), unit="epoch"):
            model.train()
            running_loss = 0.0
            running_correct = 0
            seen = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                bs = y.numel()
                running_loss += loss.item() * bs
                running_correct += (logits.argmax(1) == y).sum().item()
                seen += bs
            scheduler.step()

            # Val accuracy
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_loss_sum += criterion(logits, y).item() * y.numel()
                    val_correct += (logits.argmax(1) == y).sum().item()
                    val_total += y.numel()

            epoch_data["epoch"].append(epoch)
            epoch_data["train_loss"].append(running_loss / seen)
            epoch_data["val_loss"].append(val_loss_sum / val_total)
            epoch_data["train_accuracy"].append(running_correct / seen)
            epoch_data["val_accuracy"].append(val_correct / val_total)
            best_acc = max(best_acc, val_correct / val_total)

        histories[model_name] = epoch_data
        print(f"    Best val acc: {best_acc:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"VGG-A (lr=1e-4)": "#3066be", "VGG-A + BN (lr=1e-3)": "#d1495b"}

    for name, data in histories.items():
        color = colors[name]
        axes[0].plot(data["epoch"], data["train_loss"], color=color, label=f"{name} train")
        axes[0].plot(data["epoch"], data["val_loss"], color=color, linestyle="--", label=f"{name} val")
        axes[1].plot(data["epoch"], data["train_accuracy"], color=color, label=f"{name} train")
        axes[1].plot(data["epoch"], data["val_accuracy"], color=color, linestyle="--", label=f"{name} val")

    axes[0].set_title("Loss: VGG-A vs VGG-A + BN")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy: VGG-A vs VGG-A + BN")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "task2_performance_comparison.png", dpi=180)
    plt.close(fig)
    print(f"  Saved performance comparison plot")

    return {name: {"best_val_accuracy": max(data["val_accuracy"])}
            for name, data in histories.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="PJ2 Task 2: BN Analysis")
    parser.add_argument("--mode", choices=["all", "landscape", "gradient", "perf"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-items", type=int, default=-1)
    parser.add_argument("--val-items", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.figures_dir = args.out_dir / "figures"
    args.results_dir = args.out_dir / "results"
    for path in [args.figures_dir, args.results_dir]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    device = get_device()
    print(f"Using device: {device}")

    summary = {"device": str(device), "epochs": args.epochs}

    if args.mode in ["all", "perf"]:
        summary["performance"] = run_performance_comparison(args)

    if args.mode in ["all", "landscape"]:
        run_landscape_experiment(args)
        summary["landscape"] = "completed"

    if args.mode in ["all", "gradient"]:
        summary["gradient_analysis"] = run_gradient_analysis(args)

    summary_path = args.results_dir / "task2_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'='*60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
