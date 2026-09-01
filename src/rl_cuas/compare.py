from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from stable_baselines3 import PPO
from .config import ScenarioConfig
from .evaluate import run_episode
from .policies.baselines import ClassicThreatPolicy, RandomPolicy


def compare_policies(model_path: str, n_episodes: int, seeds: list[int], scenario: ScenarioConfig, results_dir: str):
    rows = []
    model = PPO.load(model_path)
    classic = ClassicThreatPolicy()
    random_policy = RandomPolicy()

    for seed in seeds:
        for ep in range(n_episodes):
            s = seed + ep
            rows.append(run_episode("DeepRL", model, s, scenario))
            rows.append(run_episode("Classic", classic, s, scenario))
            rows.append(run_episode("Random", random_policy, s, scenario))

    df = pd.DataFrame(rows)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "simulation_metrics.csv", index=False)

    summary = df.groupby("policy")[["damage_pct", "tracking_pct", "weapon_utilization_pct"]].mean().round(2)
    summary.to_csv(out_dir / "summary_metrics.csv")

    _plot_damage_distribution(df, out_dir / "damage_distribution.png")
    _plot_metric_histograms(df, out_dir / "tracking_utilization.png")
    _plot_correlation(df, out_dir / "correlation_analysis.png")
    return df, summary


def _apply_white_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#0F172A",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": "#0F172A",
        "font.size": 11,
    })


def _plot_damage_distribution(df, path):
    _apply_white_style()
    colors = {
        "DeepRL": "#2563EB",
        "Classic": "#64748B",
        "Random": "#06B6D4",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for name in ["DeepRL", "Classic", "Random"]:
        data = df[df["policy"] == name]["damage_pct"].to_numpy()
        if len(data) == 0:
            continue

        mu, std = norm.fit(data)
        ax.hist(
            data,
            bins=18,
            density=True,
            alpha=0.35,
            color=colors[name],
            edgecolor="white",
            linewidth=1.0,
            label=f"{name} Histogram",
        )

        xs = np.linspace(max(0, data.min() - 5), data.max() + 5, 200)
        ys = norm.pdf(xs, mu, std)
        ax.plot(
            xs,
            ys,
            color=colors[name],
            linestyle="--",
            linewidth=2.5,
            label=f"{name} Gaussian (μ={mu:.2f}, σ={std:.2f})",
        )
        ax.axvline(mu, color=colors[name], linestyle=":", linewidth=2)

    ax.set_title("Comparison of Damage Distributions", fontsize=16, fontweight="bold", color="#0F172A", pad=14)
    ax.set_xlabel("Cumulative Damage [%]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_metric_histograms(df, path):
    _apply_white_style()
    colors = {
        "DeepRL": "#2563EB",
        "Classic": "#64748B",
        "Random": "#06B6D4",
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    configs = [
        (axes[0], "tracking_pct", "Comparison of Effectors Kinematic Performance", "In-Tracking Time [%]"),
        (axes[1], "weapon_utilization_pct", "Comparison of Effectors Weapon Utilization", "Weapon Utilization [%]"),
    ]

    for ax, key, title, xlabel in configs:
        hist_handles = []
        gauss_handles = []

        for name in ["DeepRL", "Classic", "Random"]:
            data = df[df["policy"] == name][key].to_numpy()
            if len(data) == 0:
                continue

            color = colors[name]
            ax.hist(
                data,
                bins=15,
                density=True,
                alpha=0.35,
                color=color,
                edgecolor="white",
                linewidth=1.0,
            )
            hist_handles.append(
                plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.35, label=f"{name} Histogram")
            )

            mu, std = norm.fit(data)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            line, = ax.plot(
                x,
                p,
                color=color,
                linewidth=2.5,
                linestyle="--",
                label=f"{name} Gaussian (μ={mu:.2f}, σ={std:.2f})",
            )
            gauss_handles.append(line)
            ax.axvline(mu, color=color, linestyle=":", linewidth=2)

        ax.set_title(title, fontsize=15, fontweight="bold", color="#0F172A", pad=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.legend(
            handles=hist_handles + gauss_handles,
            fontsize=8,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="#CBD5E1",
        )

        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_correlation(df, path):
    _apply_white_style()

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    colors = {"DeepRL": "#2563EB", "Classic": "#64748B"}

    rl = df[df["policy"] == "DeepRL"]
    cl = df[df["policy"] == "Classic"]

    def draw(ax, y_col, title, ylabel):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=15, fontweight="bold", color="#0F172A", pad=14)
        ax.set_xlabel("Cumulative Damage [%]", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.35, color="#E2E8F0")

        ax.scatter(
            rl["damage_pct"],
            rl[y_col],
            color=colors["DeepRL"],
            edgecolor="white",
            alpha=0.85,
            label="DeepRL",
            s=60,
        )
        ax.scatter(
            cl["damage_pct"],
            cl[y_col],
            color=colors["Classic"],
            edgecolor="white",
            alpha=0.75,
            label="Classic",
            s=60,
        )

        corr = np.corrcoef(rl["damage_pct"], rl[y_col])[0, 1] if len(rl) > 1 else 0.0
        ax.text(
            0.05,
            0.92,
            f"Corr coef: {corr:.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.95, boxstyle="round,pad=0.35"),
            fontsize=10,
            color="#0F172A",
            fontweight="bold",
        )
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")

        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    draw(axs[0], "tracking_pct", "Damage vs Tracking", "In-Tracking Time [%]")
    draw(axs[1], "weapon_utilization_pct", "Damage vs Weapon Utilization", "Weapon Utilization [%]")

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()