#!/usr/bin/env python3
"""Picking-time regression with PyTorch feedforward networks.

This script implements:
1. 5-fold CV on the 80% training pool with fold-wise normalization.
2. 3-hidden-layer neural network training (32 -> 16 -> 8).
3. Final retraining on full CV pool and held-out test evaluation.
4. Linear OLS baseline on the same normalized data.
5. Predicted-vs-actual and loss-curve plots.
6. A 2-page PDF report artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
N_SPLITS = 5


def generate_picking_time_dataset(n: int = 400, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the synthetic warehouse picking-time dataset used in the exercise."""
    rng = np.random.default_rng(seed)

    distance = rng.uniform(2.0, 30.0, n)
    load = rng.uniform(1.0, 50.0, n)
    congestion = rng.poisson(1.5, n).astype(np.float32)
    battery = rng.beta(5.0, 2.0, n)
    aisle_width = rng.uniform(1.5, 3.0, n)

    y = (
        5.0
        + 0.8 * distance
        + 0.15 * load
        + 0.4 * congestion * distance
        + 12.0 * (battery < 0.2).astype(np.float32)
        - 2.0 * aisle_width
        + 0.01 * (distance**2)
        + rng.normal(0.0, 2.0, n)
    )

    x = np.column_stack([distance, load, congestion, battery, aisle_width]).astype(np.float32)
    y = y.astype(np.float32)
    feature_names = np.array(["distance", "load", "congestion", "battery", "aisle_width"], dtype=object)
    return x, y, feature_names


def ensure_dataset(path: Path, seed: int = SEED, n: int = 400) -> None:
    """Create the dataset on disk if it does not exist."""
    if path.exists():
        return

    x, y, feature_names = generate_picking_time_dataset(n=n, seed=seed)
    np.savez(path, X=x, y=y, feature_names=feature_names)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class FoldArtifacts:
    train_loss: np.ndarray
    val_loss: np.ndarray
    fold_rmse_seconds: float


@dataclass
class CrossValidationResult:
    folds: list[FoldArtifacts]
    mean_train_loss: np.ndarray
    mean_val_loss: np.ndarray
    std_val_loss: np.ndarray
    fold_rmses_seconds: np.ndarray


class PickingTimeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class NormalizationStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def compute_norm_stats(x: np.ndarray, y: np.ndarray) -> NormalizationStats:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std < 1e-8:
        y_std = 1.0

    return NormalizationStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


def normalize_xy(x: np.ndarray, y: np.ndarray, stats: NormalizationStats) -> tuple[np.ndarray, np.ndarray]:
    x_n = (x - stats.x_mean) / stats.x_std
    y_n = (y - stats.y_mean) / stats.y_std
    return x_n.astype(np.float32), y_n.astype(np.float32)


def normalize_x(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((x - stats.x_mean) / stats.x_std).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y).unsqueeze(1)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_mse(model: nn.Module, x: np.ndarray, y: np.ndarray, criterion: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y).unsqueeze(1)
        pred = model(x_t)
        loss = criterion(pred, y_t)
    return float(loss.item())


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: list[int],
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
) -> FoldArtifacts:
    model = PickingTimeNet(input_dim=x_train.shape[1], hidden_layers=hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(evaluate_mse(model, x_val, y_val, criterion))

    # Fold RMSE in original units using final model and validation split.
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(x_val)).squeeze(1).numpy()
    fold_rmse_norm = math.sqrt(float(np.mean((val_pred - y_val) ** 2)))

    return FoldArtifacts(
        train_loss=np.array(train_losses, dtype=np.float64),
        val_loss=np.array(val_losses, dtype=np.float64),
        fold_rmse_seconds=fold_rmse_norm,
    )


def run_cross_validation(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    hidden_layers: list[int],
    n_splits: int = N_SPLITS,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> CrossValidationResult:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_artifacts: list[FoldArtifacts] = []

    for train_idx, val_idx in kf.split(x_pool):
        x_tr_raw, y_tr_raw = x_pool[train_idx], y_pool[train_idx]
        x_va_raw, y_va_raw = x_pool[val_idx], y_pool[val_idx]

        stats = compute_norm_stats(x_tr_raw, y_tr_raw)
        x_tr, y_tr = normalize_xy(x_tr_raw, y_tr_raw, stats)
        x_va, y_va = normalize_xy(x_va_raw, y_va_raw, stats)

        fold = train_one_fold(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_va,
            y_val=y_va,
            hidden_layers=hidden_layers,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        fold.fold_rmse_seconds = fold.fold_rmse_seconds * stats.y_std
        fold_artifacts.append(fold)

    train_matrix = np.stack([f.train_loss for f in fold_artifacts], axis=0)
    val_matrix = np.stack([f.val_loss for f in fold_artifacts], axis=0)
    fold_rmses = np.array([f.fold_rmse_seconds for f in fold_artifacts], dtype=np.float64)

    return CrossValidationResult(
        folds=fold_artifacts,
        mean_train_loss=train_matrix.mean(axis=0),
        mean_val_loss=val_matrix.mean(axis=0),
        std_val_loss=val_matrix.std(axis=0),
        fold_rmses_seconds=fold_rmses,
    )


def retrain_on_full_pool_and_evaluate(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layers: list[int],
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
) -> tuple[float, np.ndarray]:
    stats = compute_norm_stats(x_pool, y_pool)
    x_pool_n, y_pool_n = normalize_xy(x_pool, y_pool, stats)
    x_test_n = normalize_x(x_test, stats)
    y_test_n = ((y_test - stats.y_mean) / stats.y_std).astype(np.float32)

    model = PickingTimeNet(input_dim=x_pool.shape[1], hidden_layers=hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = make_loader(x_pool_n, y_pool_n, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred_n = model(torch.from_numpy(x_test_n)).squeeze(1).numpy()

    test_pred = test_pred_n * stats.y_std + stats.y_mean
    rmse = math.sqrt(float(np.mean((test_pred - y_test) ** 2)))

    _ = y_test_n  # Kept for clarity that test normalization is applied consistently.
    return rmse, test_pred


def fit_linear_baseline(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, np.ndarray]:
    stats = compute_norm_stats(x_pool, y_pool)
    x_pool_n, y_pool_n = normalize_xy(x_pool, y_pool, stats)
    x_test_n = normalize_x(x_test, stats)

    x_pool_design = np.concatenate([np.ones((x_pool_n.shape[0], 1), dtype=np.float32), x_pool_n], axis=1)
    x_test_design = np.concatenate([np.ones((x_test_n.shape[0], 1), dtype=np.float32), x_test_n], axis=1)

    w, *_ = np.linalg.lstsq(x_pool_design, y_pool_n, rcond=None)
    pred_test_n = x_test_design @ w
    pred_test = pred_test_n * stats.y_std + stats.y_mean
    rmse = math.sqrt(float(np.mean((pred_test - y_test) ** 2)))
    return rmse, pred_test


def make_cv_plot(cv: CrossValidationResult, out_file: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    epochs_axis = np.arange(1, len(cv.mean_train_loss) + 1)

    for idx, fold in enumerate(cv.folds, start=1):
        ax.plot(epochs_axis, fold.val_loss, alpha=0.35, linewidth=1.0, label=f"Fold {idx} val")

    ax.plot(epochs_axis, cv.mean_train_loss, linewidth=2.2, label="Mean train")
    ax.plot(epochs_axis, cv.mean_val_loss, linewidth=2.2, label="Mean val")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalized, log scale)")
    ax.set_title("5-fold CV: per-fold validation and mean curves")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    return fig


def make_depth_plot(depth_curves: dict[str, np.ndarray], out_file: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    epochs_axis = np.arange(1, len(next(iter(depth_curves.values()))) + 1)

    for label, curve in depth_curves.items():
        ax.plot(epochs_axis, curve, linewidth=2.2, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean validation MSE (normalized, log scale)")
    ax.set_title("Depth experiment: mean CV validation curves")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    return fig


def make_scatter_plot(
    y_true: np.ndarray,
    y_pred_nn: np.ndarray,
    y_pred_lin: np.ndarray,
    out_file: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lo = float(min(y_true.min(), y_pred_nn.min(), y_pred_lin.min()))
    hi = float(max(y_true.max(), y_pred_nn.max(), y_pred_lin.max()))

    axes[0].scatter(y_true, y_pred_nn, alpha=0.8)
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    axes[0].set_title("Neural Network: predicted vs actual")
    axes[0].set_xlabel("Actual picking time (s)")
    axes[0].set_ylabel("Predicted picking time (s)")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(y_true, y_pred_lin, alpha=0.8, color="tab:orange")
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    axes[1].set_title("Linear OLS: predicted vs actual")
    axes[1].set_xlabel("Actual picking time (s)")
    axes[1].set_ylabel("Predicted picking time (s)")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    return fig


def infer_overfitting_statement(mean_val_curve: np.ndarray) -> str:
    best_epoch = int(np.argmin(mean_val_curve) + 1)
    tail_window = mean_val_curve[-20:]
    tail_slope = float(np.mean(np.diff(tail_window)))

    if tail_slope > 0:
        return (
            f"Validation loss reaches its minimum around epoch {best_epoch} and then trends upward "
            "toward the end, indicating overfitting after the optimum epoch."
        )

    return (
        f"Validation loss reaches its minimum around epoch {best_epoch} and then mostly plateaus, "
        "indicating limited overfitting under the current regularization and data size."
    )


def build_two_page_report(
    report_file: Path,
    cv_plot: plt.Figure,
    depth_plot: plt.Figure,
    scatter_plot: plt.Figure,
    metrics: dict[str, Any],
    overfit_note: str,
) -> None:
    with PdfPages(report_file) as pdf:
        # Page 1: performance summary and training curves.
        page1 = plt.figure(figsize=(11.69, 8.27))
        page1.suptitle("Exercise 6.1 Report: Picking-Time Neural Network", fontsize=16, y=0.98)

        ax_text = page1.add_axes([0.05, 0.57, 0.9, 0.35])
        ax_text.axis("off")
        table_lines = [
            "RMSE Comparison (held-out test set):",
            f"- Neural network (3 hidden layers): {metrics['nn_test_rmse']:.4f} s",
            f"- Linear OLS baseline: {metrics['lin_test_rmse']:.4f} s",
            f"- Improvement: {metrics['rmse_improvement']:.4f} s "
            f"({metrics['rmse_improvement_pct']:.2f}% relative)",
            "",
            "Cross-validation (3 hidden layers, 5 folds):",
            f"- Mean RMSE: {metrics['cv_rmse_mean']:.4f} s",
            f"- Std RMSE: {metrics['cv_rmse_std']:.4f} s",
            "",
            "Overfitting observation:",
            f"- {overfit_note}",
            "",
            "Depth experiment (mean CV RMSE):",
            f"- 1 hidden layer [32]: {metrics['depth_rmse']['1_hidden_32']:.4f} s",
            f"- 2 hidden layers [32,16]: {metrics['depth_rmse']['2_hidden_32_16']:.4f} s",
            f"- 3 hidden layers [32,16,8]: {metrics['depth_rmse']['3_hidden_32_16_8']:.4f} s",
        ]
        ax_text.text(0.0, 1.0, "\n".join(table_lines), va="top", fontsize=11)

        cv_img = np.asarray(cv_plot.canvas.buffer_rgba())
        depth_img = np.asarray(depth_plot.canvas.buffer_rgba())

        ax_cv = page1.add_axes([0.05, 0.05, 0.43, 0.45])
        ax_cv.imshow(cv_img)
        ax_cv.axis("off")
        ax_cv.set_title("CV curves", fontsize=10)

        ax_depth = page1.add_axes([0.52, 0.05, 0.43, 0.45])
        ax_depth.imshow(depth_img)
        ax_depth.axis("off")
        ax_depth.set_title("Depth experiment", fontsize=10)

        pdf.savefig(page1)
        plt.close(page1)

        # Page 2: predicted-vs-actual and discussion.
        page2 = plt.figure(figsize=(11.69, 8.27))
        page2.suptitle("Predicted vs Actual Analysis", fontsize=16, y=0.98)

        scatter_img = np.asarray(scatter_plot.canvas.buffer_rgba())
        ax_scatter = page2.add_axes([0.05, 0.2, 0.9, 0.7])
        ax_scatter.imshow(scatter_img)
        ax_scatter.axis("off")

        ax_note = page2.add_axes([0.05, 0.05, 0.9, 0.12])
        ax_note.axis("off")
        discussion = (
            "Discussion: The linear model underfits nonlinear interactions and tends to bias predictions "
            "at the extremes (high and low picking times), while the neural network tracks the diagonal "
            "more closely across the full range. This matches the expected limits of linearity for "
            "interaction-rich operational data."
        )
        ax_note.text(0.0, 0.95, discussion, va="top", fontsize=11)

        pdf.savefig(page2)
        plt.close(page2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Picking-time neural network experiment")
    parser.add_argument("--data", type=Path, default=Path("picking_time_data.npz"), help="Path to .npz dataset")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for plots/reports")
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("exercise_1_report.pdf"),
        help="PDF report output path",
    )
    parser.add_argument(
        "--generate-data-if-missing",
        action="store_true",
        help="Generate the synthetic dataset using the exercise formula if --data is missing",
    )
    args = parser.parse_args()

    if args.generate_data_if_missing:
        ensure_dataset(args.data, seed=SEED, n=400)

    if not args.data.exists():
        raise FileNotFoundError(
            "Dataset not found at "
            f"{args.data}. Place picking_time_data.npz in the workspace, pass --data, "
            "or run with --generate-data-if-missing."
        )

    set_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    feature_names = list(data["feature_names"])

    x_pool, x_test, y_pool, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True)

    # Main 3-hidden-layer model via 5-fold CV.
    main_hidden = [32, 16, 8]
    cv = run_cross_validation(
        x_pool=x_pool,
        y_pool=y_pool,
        hidden_layers=main_hidden,
        n_splits=N_SPLITS,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    cv_rmse_mean = float(cv.fold_rmses_seconds.mean())
    cv_rmse_std = float(cv.fold_rmses_seconds.std())

    nn_test_rmse, y_pred_nn = retrain_on_full_pool_and_evaluate(
        x_pool=x_pool,
        y_pool=y_pool,
        x_test=x_test,
        y_test=y_test,
        hidden_layers=main_hidden,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )

    lin_test_rmse, y_pred_lin = fit_linear_baseline(x_pool=x_pool, y_pool=y_pool, x_test=x_test, y_test=y_test)

    depth_specs: dict[str, list[int]] = {
        "1 hidden [32]": [32],
        "2 hidden [32,16]": [32, 16],
        "3 hidden [32,16,8]": [32, 16, 8],
    }

    depth_curves: dict[str, np.ndarray] = {}
    depth_rmse_summary: dict[str, float] = {}

    for label, layers in depth_specs.items():
        depth_cv = run_cross_validation(
            x_pool=x_pool,
            y_pool=y_pool,
            hidden_layers=layers,
            n_splits=N_SPLITS,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
        depth_curves[label] = depth_cv.mean_val_loss

        if layers == [32]:
            depth_rmse_summary["1_hidden_32"] = float(depth_cv.fold_rmses_seconds.mean())
        elif layers == [32, 16]:
            depth_rmse_summary["2_hidden_32_16"] = float(depth_cv.fold_rmses_seconds.mean())
        elif layers == [32, 16, 8]:
            depth_rmse_summary["3_hidden_32_16_8"] = float(depth_cv.fold_rmses_seconds.mean())

    cv_plot = make_cv_plot(cv, args.out_dir / "cv_loss_curves.png")
    depth_plot = make_depth_plot(depth_curves, args.out_dir / "depth_experiment_curves.png")
    scatter_plot = make_scatter_plot(y_test, y_pred_nn, y_pred_lin, args.out_dir / "pred_vs_actual_test.png")

    overfit_note = infer_overfitting_statement(cv.mean_val_loss)

    rmse_improvement = lin_test_rmse - nn_test_rmse
    rmse_improvement_pct = (rmse_improvement / lin_test_rmse * 100.0) if lin_test_rmse != 0 else 0.0

    metrics: dict[str, Any] = {
        "feature_names": feature_names,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "nn_test_rmse": float(nn_test_rmse),
        "lin_test_rmse": float(lin_test_rmse),
        "rmse_improvement": float(rmse_improvement),
        "rmse_improvement_pct": float(rmse_improvement_pct),
        "depth_rmse": depth_rmse_summary,
    }

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_path = args.report_file
    if report_path.parent != Path(""):
        report_path.parent.mkdir(parents=True, exist_ok=True)

    build_two_page_report(
        report_file=report_path,
        cv_plot=cv_plot,
        depth_plot=depth_plot,
        scatter_plot=scatter_plot,
        metrics=metrics,
        overfit_note=overfit_note,
    )

    print("=== Picking-Time Neural Network Results ===")
    print(f"Features: {feature_names}")
    print(f"CV RMSE (5-fold, mean +- std): {cv_rmse_mean:.4f} +- {cv_rmse_std:.4f} s")
    print(f"Test RMSE (NN): {nn_test_rmse:.4f} s")
    print(f"Test RMSE (Linear OLS): {lin_test_rmse:.4f} s")
    print(f"NN improvement: {rmse_improvement:.4f} s ({rmse_improvement_pct:.2f}%)")
    print(f"Overfitting note: {overfit_note}")
    print("Artifacts written to:")
    print(f"- {args.out_dir / 'cv_loss_curves.png'}")
    print(f"- {args.out_dir / 'depth_experiment_curves.png'}")
    print(f"- {args.out_dir / 'pred_vs_actual_test.png'}")
    print(f"- {report_path}")
    print(f"- {metrics_path}")


if __name__ == "__main__":
    main()
