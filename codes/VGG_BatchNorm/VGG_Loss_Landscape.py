import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light, get_number_of_parameters


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


def build_model(name):
    models = {
        "vgg_a": VGG_A,
        "vgg_a_bn": VGG_A_BatchNorm,
        "vgg_a_dropout": VGG_A_Dropout,
        "vgg_a_light": VGG_A_Light,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choices: {sorted(models)}")
    return models[name]()


@torch.no_grad()
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def get_average_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        batch_size = y.numel()
        total_loss += criterion(model(x), y).item() * batch_size
        total += batch_size
    return total_loss / total if total else 0.0


def save_sample_grid(loader, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x, y = next(iter(loader))
    x = x[:16].permute(0, 2, 3, 1).numpy()
    x = x * 0.5 + 0.5

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(np.clip(x[idx], 0, 1))
        ax.set_title(str(int(y[idx])), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_loaders(args, train=True):
    train_loader = get_cifar_loader(
        root=str(args.data_root),
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_items,
        augment=args.augment,
    )
    val_loader = get_cifar_loader(
        root=str(args.data_root),
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.val_items,
        augment=False,
    )
    return train_loader, val_loader


def train(model, optimizer, criterion, train_loader, val_loader, device, epochs_n=20, scheduler=None):
    model.to(device)
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "batch_losses": [],
        "gradient_norms": [],
    }

    best_state = None
    best_val_accuracy = 0.0

    for epoch in tqdm(range(1, epochs_n + 1), unit="epoch"):
        model.train()
        running_loss = 0.0
        seen = 0
        batch_losses = []
        gradient_norms = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            last_linear = model.classifier[-1]
            if last_linear.weight.grad is not None:
                gradient_norms.append(float(last_linear.weight.grad.norm().detach().cpu()))

            optimizer.step()

            batch_size = y.numel()
            running_loss += loss.item() * batch_size
            seen += batch_size
            batch_losses.append(float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / seen
        val_loss = get_average_loss(model, val_loader, criterion, device)
        train_accuracy = get_accuracy(model, train_loader, device)
        val_accuracy = get_accuracy(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["batch_losses"].extend(batch_losses)
        history["gradient_norms"].extend(gradient_norms)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    return history, best_state, best_val_accuracy


def save_history(history, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])
        for row in zip(
            history["epoch"],
            history["train_loss"],
            history["val_loss"],
            history["train_accuracy"],
            history["val_accuracy"],
        ):
            writer.writerow(row)


def plot_learning_curves(histories, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=f"{name} train")
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(history["epoch"], history["train_accuracy"], label=f"{name} train")
        axes[1].plot(history["epoch"], history["val_accuracy"], linestyle="--", label=f"{name} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def min_max_curve(loss_runs):
    min_len = min(len(run) for run in loss_runs if run)
    if min_len == 0:
        return [], []
    aligned = np.array([run[:min_len] for run in loss_runs])
    return aligned.min(axis=0), aligned.max(axis=0)


def plot_loss_landscape(landscape_histories, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "vgg_a": "#3066be",
        "vgg_a_bn": "#d1495b",
    }

    for model_name, lr_histories in landscape_histories.items():
        min_curve, max_curve = min_max_curve([h["batch_losses"] for h in lr_histories.values()])
        if len(min_curve) == 0:
            continue
        steps = np.arange(len(min_curve))
        color = colors.get(model_name, None)
        ax.plot(steps, min_curve, color=color, linewidth=1.4, label=f"{model_name} min")
        ax.plot(steps, max_curve, color=color, linewidth=1.4, linestyle="--", label=f"{model_name} max")
        ax.fill_between(steps, min_curve, max_curve, color=color, alpha=0.16)

    ax.set_title("Loss Landscape Proxy Across Learning Rates")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Batch loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_optimizer(name, parameters, lr, weight_decay):
    optimizers = {
        "adam": lambda: torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
        "adamw": lambda: torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay),
        "sgd": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay),
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer '{name}'. Choices: {sorted(optimizers)}")
    return optimizers[name]()


def run_single_training(args, model_name, lr, run_name, epochs, optimizer_name=None):
    device = get_device()
    set_random_seeds(args.seed, str(device))
    train_loader, val_loader = make_loaders(args)

    model = build_model(model_name)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt_name = optimizer_name or args.optimizer
    optimizer = build_optimizer(opt_name, model.parameters(), lr=lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history, best_state, best_val_accuracy = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs_n=epochs,
        scheduler=scheduler,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"{run_name}.pt"
    torch.save(
        {
            "model": model_name,
            "parameters": get_number_of_parameters(model),
            "best_val_accuracy": best_val_accuracy,
            "optimizer": opt_name,
            "state_dict": best_state,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        checkpoint_path,
    )
    save_history(history, args.results_dir / f"{run_name}.csv")
    return history, checkpoint_path, best_val_accuracy, get_number_of_parameters(model)


def run_comparison(args):
    histories = {}
    summary = {}
    for model_name in args.models:
        run_name = f"{model_name}_lr{args.lr:g}_ep{args.epochs}"
        history, checkpoint_path, best_acc, params = run_single_training(
            args=args,
            model_name=model_name,
            lr=args.lr,
            run_name=run_name,
            epochs=args.epochs,
        )
        histories[model_name] = history
        summary[model_name] = {
            "best_val_accuracy": best_acc,
            "test_error": 1.0 - best_acc,
            "parameters": params,
            "checkpoint": str(checkpoint_path),
        }

    plot_learning_curves(histories, args.figures_dir / "learning_curves.png")
    return summary


def run_landscape(args):
    landscape_histories = {}
    summary = {}
    for model_name in ["vgg_a", "vgg_a_bn"]:
        lr_histories = {}
        for lr in args.landscape_lrs:
            run_name = f"landscape_{model_name}_lr{lr:g}_ep{args.landscape_epochs}"
            history, checkpoint_path, best_acc, params = run_single_training(
                args=args,
                model_name=model_name,
                lr=lr,
                run_name=run_name,
                epochs=args.landscape_epochs,
            )
            lr_histories[f"{lr:g}"] = history
            summary[run_name] = {
                "best_val_accuracy": best_acc,
                "test_error": 1.0 - best_acc,
                "parameters": params,
                "checkpoint": str(checkpoint_path),
            }
        landscape_histories[model_name] = lr_histories

    plot_loss_landscape(landscape_histories, args.figures_dir / "loss_landscape_comparison.png")
    return summary


def run_optimizer_comparison(args):
    histories = {}
    summary = {}
    optimizers = ["adam", "adamw", "sgd"]
    model_name = "vgg_a_bn"
    for opt_name in optimizers:
        run_name = f"{model_name}_{opt_name}_lr{args.lr:g}_ep{args.epochs}"
        history, checkpoint_path, best_acc, params = run_single_training(
            args=args,
            model_name=model_name,
            lr=args.lr,
            run_name=run_name,
            epochs=args.epochs,
            optimizer_name=opt_name,
        )
        histories[opt_name] = history
        summary[opt_name] = {
            "best_val_accuracy": best_acc,
            "test_error": 1.0 - best_acc,
            "parameters": params,
            "checkpoint": str(checkpoint_path),
        }

    plot_learning_curves(histories, args.figures_dir / "optimizer_comparison.png")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="PJ2 CIFAR-10 and VGG BatchNorm experiments")
    parser.add_argument("--mode", choices=["quick", "train", "landscape", "optim", "all"], default="quick")
    parser.add_argument("--models", nargs="+", default=["vgg_a", "vgg_a_bn"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--landscape-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--landscape-lrs", nargs="+", type=float, default=[1e-3, 2e-3, 1e-4, 5e-4])
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--use-cosine-schedule", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--n-items", type=int, default=2048, help="Use -1 for full training set.")
    parser.add_argument("--val-items", type=int, default=1000, help="Use -1 for full test set.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--augment", action="store_true")
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
    print(f"Using device: {get_device()}")

    train_loader, _ = make_loaders(args)
    save_sample_grid(train_loader, args.figures_dir / "cifar10_samples.png")

    summary = {
        "device": str(get_device()),
        "data_root": str(args.data_root),
        "mode": args.mode,
    }

    if args.mode in ["quick", "train", "all"]:
        summary["training"] = run_comparison(args)

    if args.mode in ["optim", "all"]:
        summary["optimizer_comparison"] = run_optimizer_comparison(args)

    if args.mode in ["landscape", "all"]:
        summary["landscape"] = run_landscape(args)

    summary_path = args.results_dir / f"summary_{args.mode}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
