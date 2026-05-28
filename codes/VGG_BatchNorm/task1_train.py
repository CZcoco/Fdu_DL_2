"""
PJ2 Task 1: CIFAR-10 Training Experiments
- Architecture comparison (VGG_A, VGG_A_BN, VGG_A_Dropout, VGG_A_Light, ResNet_Small)
- Optimizer comparison (Adam, AdamW, SGD)
- Activation comparison (ReLU, LeakyReLU, ELU)
- Loss function comparison (CE, CE+LabelSmoothing, FocalLoss)
- Filter visualization
- Gradient norm analysis
- Training speed
"""
import argparse
import csv
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
from torch.nn import functional as F
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import (
    VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light, ResNet_Small,
    get_number_of_parameters,
)


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_model(name, activation="relu"):
    models = {
        "vgg_a": VGG_A,
        "vgg_a_bn": VGG_A_BatchNorm,
        "vgg_a_dropout": VGG_A_Dropout,
        "vgg_a_light": VGG_A_Light,
        "resnet_small": ResNet_Small,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choices: {sorted(models)}")
    return models[name](activation=activation)


def build_optimizer(name, parameters, lr, weight_decay):
    optimizers = {
        "adam": lambda: torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
        "adamw": lambda: torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay),
        "sgd": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay),
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer '{name}'. Choices: {sorted(optimizers)}")
    return optimizers[name]()


def build_criterion(name):
    if name == "ce":
        return nn.CrossEntropyLoss()
    elif name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif name == "focal":
        return FocalLoss(gamma=2.0)
    else:
        raise ValueError(f"Unknown loss '{name}'. Choices: ce, ce_ls, focal")


@torch.no_grad()
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def get_average_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        bs = y.numel()
        total_loss += criterion(model(x), y).item() * bs
        total += bs
    return total_loss / total if total else 0.0


def train(model, optimizer, criterion, train_loader, val_loader, device, epochs_n=20, scheduler=None):
    model.to(device)
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "batch_losses": [], "gradient_norms": [],
        "epoch_times": [],
    }

    best_state = None
    best_val_accuracy = 0.0

    for epoch in tqdm(range(1, epochs_n + 1), unit="epoch"):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        batch_losses = []
        gradient_norms = []
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            gradient_norms.append(total_norm ** 0.5)

            optimizer.step()

            bs = y.numel()
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(1) == y).sum().item()
            seen += bs
            batch_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0
        train_loss = running_loss / seen
        train_accuracy = running_correct / seen
        val_accuracy = get_accuracy(model, val_loader, device)
        val_loss = get_average_loss(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["batch_losses"].extend(batch_losses)
        history["gradient_norms"].extend(gradient_norms)
        history["epoch_times"].append(epoch_time)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return history, best_state, best_val_accuracy


def save_history(history, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "epoch_time"])
        for row in zip(
            history["epoch"], history["train_loss"], history["val_loss"],
            history["train_accuracy"], history["val_accuracy"], history["epoch_times"],
        ):
            writer.writerow(row)


def plot_learning_curves(histories, output_path, title_prefix=""):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=f"{name} train")
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(history["epoch"], history["train_accuracy"], label=f"{name} train")
        axes[1].plot(history["epoch"], history["val_accuracy"], linestyle="--", label=f"{name} val")

    axes[0].set_title(f"{title_prefix}Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title(f"{title_prefix}Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gradient_norms(histories, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, history in histories.items():
        norms = history["gradient_norms"]
        # subsample for readability
        step = max(1, len(norms) // 500)
        ax.plot(range(0, len(norms), step), norms[::step], label=name, linewidth=0.8)
    ax.set_title("Gradient Norm During Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("L2 norm of gradients")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_speed(histories, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(histories.keys())
    avg_times = [np.mean(h["epoch_times"]) for h in histories.values()]
    bars = ax.bar(names, avg_times, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    ax.set_ylabel("Avg seconds per epoch")
    ax.set_title("Training Speed Comparison")
    for bar, t in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{t:.1f}s", ha='center', fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_filter_visualization(model, output_path):
    """Visualize first conv layer filters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    if first_conv is None:
        return

    weights = first_conv.weight.detach().cpu()
    n_filters = min(weights.shape[0], 64)
    ncols = 8
    nrows = (n_filters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    for idx, ax in enumerate(axes.ravel()):
        if idx < n_filters:
            w = weights[idx]
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            if w.shape[0] == 3:
                ax.imshow(w.permute(1, 2, 0).numpy())
            else:
                ax.imshow(w[0].numpy(), cmap='gray')
        ax.axis("off")
    fig.suptitle("First Conv Layer Filters", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_sample_grid(loader, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x, y = next(iter(loader))
    x = x[:16].permute(0, 2, 3, 1).numpy()
    x = x * 0.5 + 0.5

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(np.clip(x[idx], 0, 1))
        ax.set_title(classes[int(y[idx])], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_loaders(args):
    train_loader = get_cifar_loader(
        root=str(args.data_root), batch_size=args.batch_size,
        train=True, shuffle=True, num_workers=args.num_workers,
        n_items=args.n_items, augment=args.augment,
    )
    val_loader = get_cifar_loader(
        root=str(args.data_root), batch_size=args.batch_size,
        train=False, shuffle=False, num_workers=args.num_workers,
        n_items=args.val_items, augment=False,
    )
    return train_loader, val_loader


def run_single(args, model_name, lr, optimizer_name, loss_name, activation, epochs, run_name):
    device = get_device()
    set_random_seeds(args.seed, str(device))
    train_loader, val_loader = make_loaders(args)

    model = build_model(model_name, activation=activation)
    criterion = build_criterion(loss_name)
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history, best_state, best_val_accuracy = train(
        model=model, optimizer=optimizer, criterion=criterion,
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs_n=epochs, scheduler=scheduler,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"{run_name}.pt"
    torch.save({
        "model": model_name, "activation": activation,
        "parameters": get_number_of_parameters(model),
        "best_val_accuracy": best_val_accuracy,
        "optimizer": optimizer_name, "loss": loss_name,
        "state_dict": best_state,
    }, checkpoint_path)
    save_history(history, args.results_dir / f"{run_name}.csv")

    return history, model, best_val_accuracy, get_number_of_parameters(model)


def experiment_architectures(args):
    """Compare different architectures."""
    print("\n=== Experiment: Architecture Comparison ===")
    # (model, lr, optimizer) - SGD for non-BN deep nets to help convergence
    configs = [
        ("vgg_a", 0.01, "sgd"),
        ("vgg_a_bn", 1e-3, "adam"),
        ("vgg_a_dropout", 0.01, "sgd"),
        ("vgg_a_light", 1e-3, "adam"),
        ("resnet_small", 1e-3, "adam"),
    ]
    histories = {}
    summary = {}
    best_model = None
    best_acc = 0

    for model_name, lr, opt in configs:
        run_name = f"arch_{model_name}"
        print(f"  Training {model_name} (lr={lr}, opt={opt})...")
        history, model, acc, params = run_single(
            args, model_name, lr, opt, "ce", "relu", args.epochs, run_name
        )
        histories[model_name] = history
        summary[model_name] = {
            "best_val_accuracy": acc, "test_error": 1.0 - acc,
            "parameters": params, "avg_epoch_time": np.mean(history["epoch_times"]),
        }
        if acc > best_acc:
            best_acc = acc
            best_model = model

    plot_learning_curves(histories, args.figures_dir / "task1_arch_curves.png", "Architecture: ")
    plot_gradient_norms(histories, args.figures_dir / "task1_arch_gradients.png")
    plot_training_speed(histories, args.figures_dir / "task1_arch_speed.png")
    if best_model is not None:
        plot_filter_visualization(best_model, args.figures_dir / "task1_filter_vis.png")
    return summary


def experiment_optimizers(args):
    """Compare different optimizers on vgg_a_bn."""
    print("\n=== Experiment: Optimizer Comparison ===")
    configs = [
        ("adam", 1e-3),
        ("adamw", 1e-3),
        ("sgd", 0.01),
    ]
    histories = {}
    summary = {}

    for opt_name, lr in configs:
        run_name = f"optim_{opt_name}"
        print(f"  Training vgg_a_bn with {opt_name} (lr={lr})...")
        history, _, acc, params = run_single(
            args, "vgg_a_bn", lr, opt_name, "ce", "relu", args.epochs, run_name
        )
        histories[opt_name] = history
        summary[opt_name] = {
            "best_val_accuracy": acc, "test_error": 1.0 - acc,
            "parameters": params, "avg_epoch_time": np.mean(history["epoch_times"]),
        }

    plot_learning_curves(histories, args.figures_dir / "task1_optim_curves.png", "Optimizer: ")
    plot_gradient_norms(histories, args.figures_dir / "task1_optim_gradients.png")
    return summary


def experiment_activations(args):
    """Compare different activation functions on vgg_a_bn."""
    print("\n=== Experiment: Activation Comparison ===")
    activations = ["relu", "leaky_relu", "elu"]
    histories = {}
    summary = {}

    for act in activations:
        run_name = f"act_{act}"
        print(f"  Training vgg_a_bn with {act}...")
        history, _, acc, params = run_single(
            args, "vgg_a_bn", 1e-3, "adam", "ce", act, args.epochs, run_name
        )
        histories[act] = history
        summary[act] = {
            "best_val_accuracy": acc, "test_error": 1.0 - acc,
            "parameters": params, "avg_epoch_time": np.mean(history["epoch_times"]),
        }

    plot_learning_curves(histories, args.figures_dir / "task1_act_curves.png", "Activation: ")
    plot_gradient_norms(histories, args.figures_dir / "task1_act_gradients.png")
    return summary


def experiment_losses(args):
    """Compare different loss functions on vgg_a_bn."""
    print("\n=== Experiment: Loss Function Comparison ===")
    losses = [("ce", "CrossEntropy"), ("ce_ls", "CE+LabelSmooth"), ("focal", "FocalLoss")]
    histories = {}
    summary = {}

    for loss_name, display_name in losses:
        run_name = f"loss_{loss_name}"
        print(f"  Training vgg_a_bn with {display_name}...")
        history, _, acc, params = run_single(
            args, "vgg_a_bn", 1e-3, "adam", loss_name, "relu", args.epochs, run_name
        )
        histories[display_name] = history
        summary[display_name] = {
            "best_val_accuracy": acc, "test_error": 1.0 - acc,
            "parameters": params, "avg_epoch_time": np.mean(history["epoch_times"]),
        }

    plot_learning_curves(histories, args.figures_dir / "task1_loss_curves.png", "Loss: ")
    return summary


def experiment_bn_advantage(args):
    """Show BN allows higher learning rate."""
    print("\n=== Experiment: BN Advantage (High LR) ===")
    configs = [
        ("vgg_a", "VGG-A (no BN)"),
        ("vgg_a_bn", "VGG-A + BN"),
    ]
    histories = {}
    summary = {}
    lr = 1e-3

    for model_name, display_name in configs:
        run_name = f"bn_adv_{model_name}_lr{lr:g}"
        print(f"  Training {display_name} at lr={lr}...")
        history, _, acc, params = run_single(
            args, model_name, lr, "adam", "ce", "relu", args.epochs, run_name
        )
        histories[display_name] = history
        summary[display_name] = {
            "best_val_accuracy": acc, "test_error": 1.0 - acc, "parameters": params,
        }

    plot_learning_curves(histories, args.figures_dir / "task1_bn_advantage.png", "BN Advantage (lr=1e-3): ")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="PJ2 Task 1: CIFAR-10 Experiments")
    parser.add_argument("--mode", choices=["all", "arch", "optim", "act", "loss", "bn"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--use-cosine-schedule", action="store_true")
    parser.add_argument("--n-items", type=int, default=-1)
    parser.add_argument("--val-items", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.figures_dir = args.out_dir / "figures"
    args.models_dir = args.out_dir / "models"
    args.results_dir = args.out_dir / "results"
    for path in [args.figures_dir, args.models_dir, args.results_dir]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    device = get_device()
    print(f"Using device: {device}")

    train_loader, _ = make_loaders(args)
    save_sample_grid(train_loader, args.figures_dir / "cifar10_samples.png")

    summary = {"device": str(device), "epochs": args.epochs, "mode": args.mode}

    if args.mode in ["all", "arch"]:
        summary["architectures"] = experiment_architectures(args)

    if args.mode in ["all", "optim"]:
        summary["optimizers"] = experiment_optimizers(args)

    if args.mode in ["all", "act"]:
        summary["activations"] = experiment_activations(args)

    if args.mode in ["all", "loss"]:
        summary["losses"] = experiment_losses(args)

    if args.mode in ["all", "bn"]:
        summary["bn_advantage"] = experiment_bn_advantage(args)

    summary_path = args.results_dir / "task1_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
