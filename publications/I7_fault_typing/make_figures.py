"""Generate the I7 paper figures from the committed benchmark CSVs.

Reads ``examples/benchmarks/i7_*.csv`` (the reproducible experiment
artifacts) and writes vector PDFs into ``figures/`` next to this script.
No data is hard-coded -- every figure is derived from the CSVs so it
stays in sync with the experiments.

Run::

    uv run python publications/I7_fault_typing/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "examples" / "benchmarks"
FIGS = Path(__file__).resolve().parent / "figures"

plt.rcParams.update(
    {
        "figure.figsize": (5.0, 3.4),
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)


def fig_operating_curve() -> None:
    """Intel-Lab: healthy FP and real-fault detection vs mean_threshold."""
    df = pd.read_csv(BENCH / "i7_intel_lab_operating_points.csv")
    fig, ax = plt.subplots()
    ax.plot(
        df["mean_threshold"],
        df["healthy_fp_rate"],
        "o-",
        color="tab:red",
        label="healthy false-positive rate",
    )
    ax.plot(
        df["mean_threshold"],
        df["fault_detect_rate"],
        "s-",
        color="tab:blue",
        label="real-fault detection rate",
    )
    ax.set_xlabel(r"mean threshold (conditional $\sigma$)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right")
    ax.set_title("Intel-Lab battery-depletion faults (online adaptive)")
    fig.savefig(FIGS / "fig_operating_curve.pdf")
    plt.close(fig)


def fig_bias_vs_sigma() -> None:
    """Bias recall vs conditional sigma, with Wilson 95% CIs."""
    df = pd.read_csv(BENCH / "i7_scaled_bias_by_channel.csv")
    df = df.sort_values("cond_sigma")
    lo = df["bias_recall"] - df["ci_lo"]
    hi = df["ci_hi"] - df["bias_recall"]
    fig, ax = plt.subplots()
    ax.errorbar(
        df["cond_sigma"],
        df["bias_recall"],
        yerr=[lo, hi],
        fmt="o",
        color="tab:purple",
        capsize=3,
    )
    for _, r in df.iterrows():
        ax.annotate(
            r["signal"],
            (r["cond_sigma"], r["bias_recall"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel(r"conditional $\sigma$ (coupling proxy)")
    ax.set_ylabel("bias recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Bias absorption rises with conditional coupling")
    fig.savefig(FIGS / "fig_bias_vs_sigma.pdf")
    plt.close(fig)


def fig_regime_fp() -> None:
    """Regime false positives vs correlation: mechanism 1 vs mechanism 2."""
    df = pd.read_csv(BENCH / "i7_regime_fp.csv")
    fig, ax = plt.subplots()
    for shift, style in ((6.0, "o-"), (10.0, "s-")):
        sub = df[(df["shift"] == shift) & (df["suppress_scale"] == 1.0)]
        sub = sub.sort_values("rho")
        ax.plot(
            sub["rho"],
            sub["fp_rate_mean"],
            style,
            label=f"shift {shift:.0f}$\\sigma$, no suppression",
        )
    # Mechanism 2 (default suppression) collapses every cell to zero.
    ax.axhline(
        0.0,
        color="tab:green",
        ls="--",
        label="default suppression (all cells)",
    )
    ax.set_xlabel(r"inter-signal correlation $\rho$")
    ax.set_ylabel("false-positive rate during regime change")
    ax.set_ylim(-0.02, 0.25)
    ax.legend()
    ax.set_title(r"Residual cancellation fails at low $\rho$")
    fig.savefig(FIGS / "fig_regime_fp.pdf")
    plt.close(fig)


def fig_deadband() -> None:
    """Co-occurring-fault magnitude at detection vs suppression scale."""
    df = pd.read_csv(BENCH / "i7_deadband.csv")
    labels = [str(s) for s in df["suppress_scale"]]
    mags = df["magnitude_sigma_at_detection"].fillna(0.0)
    detected = df["detect_rate"] > 0
    fig, ax = plt.subplots()
    colors = ["tab:blue" if d else "tab:gray" for d in detected]
    bars = ax.bar(labels, mags, color=colors)
    for b, d in zip(bars, detected, strict=True):
        if not d:
            ax.text(
                b.get_x() + b.get_width() / 2,
                0.5,
                "never\ndetected",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xlabel("suppression scale")
    ax.set_ylabel(r"fault magnitude at detection (conditional $\sigma$)")
    ax.set_title("Co-occurring-fault dead-band grows with suppression")
    fig.savefig(FIGS / "fig_deadband.pdf")
    plt.close(fig)


def main() -> None:
    """Write all figures."""
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_operating_curve()
    fig_bias_vs_sigma()
    fig_regime_fp()
    fig_deadband()
    print(f"figures written to {FIGS}")  # noqa: T201


if __name__ == "__main__":
    main()
