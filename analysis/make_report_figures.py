"""Render the reproduction report's evidence-first figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER = {"PushT": 96.0, "TwoRoom": 87.0, "Reacher": 86.0, "Cube": 74.0}


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return 100 * (center - margin), 100 * (center + margin)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#fbfaf7",
            "axes.edgecolor": "#cbc8c0",
            "axes.labelcolor": "#2d3440",
            "text.color": "#2d3440",
            "xtick.color": "#596273",
            "ytick.color": "#596273",
        }
    )


def control_figure(metrics: dict, output: Path) -> None:
    envs = list(PAPER)
    y = np.arange(len(envs))
    reproduced = [metrics["reproduced"].get(env) for env in envs]
    trained = [metrics.get("trained", {}).get(env) for env in envs]
    fig, ax = plt.subplots(figsize=(10.5, 4.7), constrained_layout=True)
    ax.barh(y + 0.25, [PAPER[e] for e in envs], height=0.22, color="#a9b0ba", label="Paper")
    for i, row in enumerate(reproduced):
        if not row:
            ax.text(2, i - 0.17, "running", va="center", color="#7a8290", fontstyle="italic")
            continue
        value = row["success_rate"]
        low, high = wilson(row["successes"], row["trials"])
        ax.barh(i, value, height=0.22, color="#345cdb", label="Released checkpoint" if i == 0 else None)
        ax.errorbar(value, i, xerr=[[value - low], [high - value]], fmt="none", ecolor="#163591", capsize=3)
        ax.text(value + 1.1, i, f"{row['successes']}/{row['trials']}  ({value:.0f}%)", va="center", fontweight="bold")
    for i, row in enumerate(trained):
        if not row:
            continue
        value = row["success_rate"]
        low, high = wilson(row["successes"], row["trials"])
        ax.barh(i - 0.25, value, height=0.22, color="#3a9d8f", label="From scratch" if i == 1 else None)
        ax.errorbar(value, i - 0.25, xerr=[[value - low], [high - value]], fmt="none", ecolor="#226b61", capsize=3)
        ax.text(value + 1.1, i - 0.25, f"{row['successes']}/{row['trials']}  ({value:.0f}%)", va="center", fontweight="bold")
    ax.set(yticks=y, yticklabels=envs, xlim=(0, 108), xlabel="Goal-conditioned control success (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dedbd4", linewidth=0.7)
    ax.set_title("Paper vs released-checkpoint vs from-scratch control", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def diagnostics_figure(metrics: dict, output: Path) -> None:
    rows = metrics["reproduced"]
    envs = [e for e in ("PushT", "TwoRoom", "Reacher", "Cube") if e in rows]
    colors = ["#345cdb", "#e7834a", "#3a9d8f", "#9b66c7"][: len(envs)]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    cov = [rows[e]["covariance_diag_mean"] for e in envs]
    rank = [rows[e]["effective_rank"] / 192 * 100 for e in envs]
    probe_linear = [rows[e]["linear_probe_r"] for e in envs]
    probe_mlp = [rows[e]["mlp_probe_r"] for e in envs]
    x = np.arange(len(envs))

    axes[0].bar(x, cov, color=colors, width=0.62)
    axes[0].axhline(1.0, color="#303642", linewidth=1.3, linestyle="--", label="SIGReg target")
    axes[0].set(xticks=x, xticklabels=envs, ylabel="Mean covariance diagonal", ylim=(0, 1.12), title="Gaussianized scale")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, rank, color=colors, width=0.62)
    axes[1].set(xticks=x, xticklabels=envs, ylabel="Effective rank (% of 192-D)", ylim=(0, 100), title="Active latent dimensions")

    width = 0.34
    axes[2].bar(x - width / 2, probe_linear, width, color="#6d7888", label="Linear")
    axes[2].bar(x + width / 2, probe_mlp, width, color="#345cdb", label="MLP")
    axes[2].set(xticks=x, xticklabels=envs, ylabel="Mean Pearson r", ylim=(0, 1.08), title="Physical-state recoverability")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#dedbd4", linewidth=0.7)
    fig.suptitle("What the learned world-model latent contains", x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def surprise_figure(metrics: dict, output: Path) -> None:
    rows = metrics["reproduced"]
    envs = [e for e in ("PushT", "TwoRoom", "Reacher", "Cube") if e in rows]
    x = np.arange(len(envs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 4.7), constrained_layout=True)
    series = [
        ("Normal transition", "normal_surprise", "#7b8492"),
        ("Visual proxy", "visual_surprise", "#e6a044"),
        ("Continuity violation", "teleport_surprise", "#c94f6d"),
    ]
    for offset, (label, key, color) in zip((-width, 0, width), series):
        vals = [rows[e][key] for e in envs]
        ax.bar(x + offset, vals, width, color=color, label=label)
    ax.set(xticks=x, xticklabels=envs, ylabel="Next-latent prediction MSE", title="Offline surprise separates ordinary and intervened observations")
    ax.grid(axis="y", color="#dedbd4", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def epoch1_ablation_figure(metrics: dict, output: Path) -> None:
    snapshot = metrics["epoch1_ablation_snapshot"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    specs = [
        ("sigreg_weight", "SIGReg weight λ", "#345cdb"),
        ("predictor_dropout", "Predictor dropout", "#e7834a"),
    ]
    for col, (key, xlabel, color) in enumerate(specs):
        rows = snapshot[key]
        labels = [f"{row['value']:g}" for row in rows]
        x = np.arange(len(rows))
        pred = [row["prediction_loss"] for row in rows]
        sigreg = [row["sigreg_loss"] for row in rows]
        axes[0, col].plot(x, pred, marker="o", linewidth=2.2, color=color)
        axes[0, col].set(xlabel=xlabel, ylabel="Validation prediction loss", title=f"Prediction vs {xlabel.lower()}")
        axes[1, col].plot(x, sigreg, marker="s", linewidth=2.2, color=color)
        axes[1, col].set(xlabel=xlabel, ylabel="Validation SIGReg diagnostic", title=f"Gaussianization vs {xlabel.lower()}")
        for ax in axes[:, col]:
            ax.grid(color="#dedbd4", linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
    axes[0, 0].set_yscale("log")
    fig.suptitle("After one epoch: regularization trade-offs", x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def collapse_trajectory_figure(metrics: dict, output: Path) -> None:
    rows = metrics["epoch1_ablation_snapshot"]["lambda_zero_trajectory"]
    epochs = [row["epoch"] for row in rows]
    pred = [row["prediction_loss"] for row in rows]
    sigreg = [row["sigreg_loss"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    ax.plot(epochs, pred, marker="o", linewidth=2.5, color="#345cdb", label="Prediction loss")
    ax.plot(epochs, sigreg, marker="s", linewidth=2.5, color="#c94f6d", label="SIGReg diagnostic")
    ax.set(
        xticks=epochs,
        xlabel="Completed epoch (0 = initialization)",
        ylabel="Held-out validation value (log scale)",
        yscale="log",
        title="Prediction-only training collapses instead of learning world state",
    )
    ax.grid(color="#dedbd4", linewidth=0.7)
    ax.legend(frameon=False, loc="center right")
    ax.annotate(
        "prediction error falls 12,826×",
        xy=(3, pred[-1]),
        xytext=(1.15, 0.0015),
        arrowprops={"arrowstyle": "->", "color": "#596273"},
        color="#2d3440",
    )
    ax.annotate("latent regularity stays at 51.49", xy=(2.9, sigreg[-1]), xytext=(1.2, 18), color="#2d3440")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("analysis/reproduction_metrics.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text())
    style()
    control_figure(metrics, args.output / "control-reproduction.png")
    diagnostics_figure(metrics, args.output / "latent-diagnostics.png")
    surprise_figure(metrics, args.output / "offline-surprise.png")
    epoch1_ablation_figure(metrics, args.output / "epoch1-ablation.png")
    collapse_trajectory_figure(metrics, args.output / "collapse-trajectory.png")


if __name__ == "__main__":
    main()
