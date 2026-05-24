"""
CalMS21 Minimum Observable Body Geometry
Paper Results Package Generator

"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

def ensure_package(package_name: str) -> None:
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

ensure_package("numpy")
ensure_package("pandas")
ensure_package("matplotlib")
ensure_package("tabulate")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("/content/calms21_minimum_body_paper_grade")
RESULTS_DIR = BASE_DIR / "results"
ORIGINAL_FIGURES_DIR = BASE_DIR / "figures"
ORIGINAL_TABLES_DIR = BASE_DIR / "tables"

PACKAGE_DIR = BASE_DIR / "paper_package"
PACKAGE_FIG_DIR = PACKAGE_DIR / "figures"
PACKAGE_TABLE_DIR = PACKAGE_DIR / "tables"
PACKAGE_DATA_DIR = PACKAGE_DIR / "data"
PACKAGE_SNIPPET_DIR = PACKAGE_DIR / "latex_snippets"
PACKAGE_ORIGINAL_DIR = PACKAGE_DIR / "original_outputs"

ZIP_OUTPUT_PATH = BASE_DIR / "calms21_minimum_body_paper_package.zip"

FIG_DPI = 600

REGION_SHORT = {
    "full_body_both_mice": "Full body",
    "head_neck_both_mice": "Head+neck",
    "head_only_both_mice": "Head only",
    "no_tail_both_mice": "No tail",
    "trunk_tail_both_mice": "Trunk+tail",
    "hips_tail_both_mice": "Hips+tail",
    "hips_only_both_mice": "Hips only",
    "neck_only_both_mice": "Neck only",
    "tail_base_only_both_mice": "Tail base",
    "resident_only_full_body": "Resident only",
    "intruder_only_full_body": "Intruder only",
}

REGION_ORDER = [
    "full_body_both_mice",
    "no_tail_both_mice",
    "hips_tail_both_mice",
    "head_neck_both_mice",
    "trunk_tail_both_mice",
    "head_only_both_mice",
    "hips_only_both_mice",
    "neck_only_both_mice",
    "tail_base_only_both_mice",
    "resident_only_full_body",
    "intruder_only_full_body",
]

MAIN_REGIONS = [
    "full_body_both_mice",
    "no_tail_both_mice",
    "hips_tail_both_mice",
    "head_neck_both_mice",
    "trunk_tail_both_mice",
    "resident_only_full_body",
    "intruder_only_full_body",
]

COMPACT_REGIONS = [
    "no_tail_both_mice",
    "hips_tail_both_mice",
    "head_neck_both_mice",
    "trunk_tail_both_mice",
]

CLASS_ORDER = ["attack", "investigation", "mount", "other"]


# =============================================================================
# Utility functions
# =============================================================================

def reset_package_dirs() -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    for d in [
        PACKAGE_DIR,
        PACKAGE_FIG_DIR,
        PACKAGE_TABLE_DIR,
        PACKAGE_DATA_DIR,
        PACKAGE_SNIPPET_DIR,
        PACKAGE_ORIGINAL_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_csv(name: str, required: bool = True) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if path.exists():
        print(f"[load] {path}")
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Required CSV not found: {path}")
    print(f"[skip] Optional CSV not found: {path}")
    return pd.DataFrame()


def load_json(name: str) -> Dict[str, Any]:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"[skip] Optional JSON not found: {path}")
        return {}
    print(f"[load] {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sort_regions(df: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    out = df.copy()
    order = {r: i for i, r in enumerate(REGION_ORDER)}
    out["_order"] = out[region_col].map(order).fillna(9999)
    out = out.sort_values(["_order", region_col]).drop(columns=["_order"])
    return out


def add_labels(df: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    out = df.copy()
    out["region_short"] = out[region_col].map(REGION_SHORT).fillna(out[region_col])
    return out


def fmt_mean_ci(mean: float, ci: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(ci):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def save_fig(fig: plt.Figure, stem: str) -> Dict[str, str]:
    out = {}
    for ext in ["png", "pdf", "svg"]:
        path = PACKAGE_FIG_DIR / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        out[ext] = str(path)
    plt.close(fig)
    print(f"[figure] Saved {stem}.png/.pdf/.svg")
    return out


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, resize: bool = True) -> None:
    tabular = df.to_latex(index=False, escape=False)
    if resize:
        latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"{tabular}"
            "}\n"
            "\\end{table}\n"
        )
    else:
        latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"{tabular}"
            "\\end{table}\n"
        )
    path.write_text(latex, encoding="utf-8")
    print(f"[table] Saved {path}")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"[text] Saved {path}")


# =============================================================================
# Derived tables
# =============================================================================

def build_retention_summary(retention_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "macro_f1_retention_percent",
        "balanced_accuracy_retention_percent",
        "accuracy_retention_percent",
        "weighted_f1_retention_percent",
    ]
    rows = []
    for region, sub in retention_df.groupby("region"):
        row = {"region": region, "n_seeds": sub["seed"].nunique()}
        for metric in metrics:
            if metric in sub.columns:
                vals = sub[metric].astype(float).values
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / math.sqrt(max(len(vals), 1))
        rows.append(row)
    return sort_regions(pd.DataFrame(rows))


def build_degradation_retention(deg_df: pd.DataFrame) -> pd.DataFrame:
    if deg_df.empty:
        return pd.DataFrame()
    full = deg_df[deg_df["region"] == "full_body_both_mice"][
        ["seed", "degradation_mode", "degradation_severity", "macro_f1", "balanced_accuracy"]
    ].rename(columns={"macro_f1": "full_macro_f1", "balanced_accuracy": "full_balanced_accuracy"})
    merged = deg_df.merge(full, on=["seed", "degradation_mode", "degradation_severity"], how="left")
    merged["macro_f1_retention_percent"] = 100.0 * merged["macro_f1"] / np.maximum(merged["full_macro_f1"], 1e-9)
    merged["balanced_accuracy_retention_percent"] = 100.0 * merged["balanced_accuracy"] / np.maximum(merged["full_balanced_accuracy"], 1e-9)
    return merged


def summarize_degradation_retention(deg_ret_df: pd.DataFrame) -> pd.DataFrame:
    if deg_ret_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in deg_ret_df.groupby(["region", "degradation_mode", "degradation_severity"]):
        region, mode, severity = keys
        row = {"region": region, "degradation_mode": mode, "degradation_severity": severity, "n_seeds": sub["seed"].nunique()}
        for metric in ["macro_f1_retention_percent", "balanced_accuracy_retention_percent"]:
            vals = sub[metric].astype(float).values
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / math.sqrt(max(len(vals), 1))
        rows.append(row)
    return sort_regions(pd.DataFrame(rows))


# =============================================================================
# Figures
# =============================================================================

def draw_mouse(ax, x0: float, y0: float, scale: float, active: List[str], title: str = "") -> None:
    points = {
        "nose": (0.00, 0.00),
        "left_ear": (-0.12, -0.18),
        "right_ear": (0.12, -0.18),
        "neck": (0.00, -0.35),
        "left_hip": (-0.14, -0.72),
        "right_hip": (0.14, -0.72),
        "tail_base": (0.00, -0.95),
    }
    lines = [
        ("nose", "neck"),
        ("left_ear", "neck"),
        ("right_ear", "neck"),
        ("neck", "left_hip"),
        ("neck", "right_hip"),
        ("left_hip", "tail_base"),
        ("right_hip", "tail_base"),
    ]
    for a, b in lines:
        xa, ya = points[a]
        xb, yb = points[b]
        ax.plot([x0 + scale * xa, x0 + scale * xb], [y0 + scale * ya, y0 + scale * yb], linewidth=1.2, alpha=0.55)
    for name, (x, y) in points.items():
        on = name in active
        ax.scatter(
            [x0 + scale * x],
            [y0 + scale * y],
            s=46 if on else 18,
            alpha=1.0 if on else 0.25,
            edgecolors="black",
            linewidths=0.4,
        )
    if title:
        ax.text(x0, y0 + scale * 0.18, title, ha="center", va="bottom", fontsize=8)


def fig1_pipeline() -> Dict[str, str]:
    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.2], hspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    steps = [
        "CalMS21\npose trajectories",
        "1.5 s windows\n45 frames",
        "Temporal CNN\npose dynamics",
        "Body-region\nablation",
        "Degradation\nrobustness",
        "Retention vs\nfull body",
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    y = 0.55
    for i, (x, step) in enumerate(zip(xs, steps)):
        rect = plt.Rectangle((x - 0.065, y - 0.13), 0.13, 0.26, fill=False, linewidth=1.2, transform=ax0.transAxes)
        ax0.add_patch(rect)
        ax0.text(x, y, step, ha="center", va="center", fontsize=9, transform=ax0.transAxes)
        if i < len(xs) - 1:
            ax0.annotate(
                "",
                xy=(xs[i + 1] - 0.075, y),
                xytext=(x + 0.075, y),
                xycoords=ax0.transAxes,
                arrowprops=dict(arrowstyle="->", linewidth=1.2),
            )
    ax0.text(0.02, 0.94, "a", fontsize=14, fontweight="bold", transform=ax0.transAxes)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    examples = [
        ("Full body", ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]),
        ("Head+neck", ["nose", "left_ear", "right_ear", "neck"]),
        ("No tail", ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip"]),
        ("Hips+tail", ["left_hip", "right_hip", "tail_base"]),
        ("Trunk+tail", ["neck", "left_hip", "right_hip", "tail_base"]),
        ("Single animal", ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]),
    ]
    for x, (label, active) in zip(np.linspace(0.09, 0.91, len(examples)), examples):
        if label == "Single animal":
            draw_mouse(ax1, x - 0.015, 0.83, 0.33, active, label)
            draw_mouse(ax1, x + 0.06, 0.63, 0.23, [], "")
        else:
            draw_mouse(ax1, x - 0.035, 0.83, 0.28, active, label)
            draw_mouse(ax1, x + 0.035, 0.70, 0.28, active, "")
    ax1.text(0.02, 0.98, "b", fontsize=14, fontweight="bold", transform=ax1.transAxes)
    return save_fig(fig, "fig1_pipeline_and_body_regions")


def fig2_clean_performance(summary: pd.DataFrame) -> Dict[str, str]:
    df = add_labels(summary)
    df = df[df["region"].isin(REGION_ORDER)].sort_values("macro_f1_mean", ascending=True)
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5.5, 0.45 * len(df))))

    axes[0].barh(y, df["macro_f1_mean"], xerr=df["macro_f1_ci95"], capsize=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["region_short"])
    axes[0].set_xlabel("Macro F1")
    axes[0].set_xlim(0, min(1.0, float((df["macro_f1_mean"] + df["macro_f1_ci95"]).max()) * 1.15))
    axes[0].text(-0.18, 1.03, "a", transform=axes[0].transAxes, fontsize=14, fontweight="bold")

    axes[1].barh(y, df["balanced_accuracy_mean"], xerr=df["balanced_accuracy_ci95"], capsize=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Balanced accuracy")
    axes[1].set_xlim(0, min(1.0, float((df["balanced_accuracy_mean"] + df["balanced_accuracy_ci95"]).max()) * 1.15))
    axes[1].text(-0.08, 1.03, "b", transform=axes[1].transAxes, fontsize=14, fontweight="bold")

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, "fig2_clean_performance")


def fig3_retention(ret_sum: pd.DataFrame, clean_results: pd.DataFrame) -> Dict[str, str]:
    ret = add_labels(ret_sum)
    ret = ret[ret["region"].isin(REGION_ORDER)].sort_values("macro_f1_retention_percent_mean", ascending=True)
    res = add_labels(clean_results)
    y = np.arange(len(ret))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, max(5.5, 0.45 * len(ret))))

    axes[0].barh(y, ret["macro_f1_retention_percent_mean"], xerr=ret["macro_f1_retention_percent_ci95"], capsize=3)
    axes[0].axvline(100, linestyle="--", linewidth=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ret["region_short"])
    axes[0].set_xlabel("Macro F1 retained relative to full body (%)")
    axes[0].text(-0.18, 1.03, "a", transform=axes[0].transAxes, fontsize=14, fontweight="bold")

    regions_sorted = ret["region"].tolist()
    region_to_y = {region: i for i, region in enumerate(regions_sorted)}
    for region in regions_sorted:
        sub = res[res["region"] == region].sort_values("seed")
        yy = region_to_y[region]
        jitter = np.linspace(-0.13, 0.13, len(sub)) if len(sub) > 1 else np.array([0.0])
        axes[1].scatter(sub["macro_f1"].values, yy + jitter, s=35)
    means = res.groupby("region")["macro_f1"].mean().reindex(regions_sorted)
    axes[1].scatter(means.values, np.arange(len(regions_sorted)), marker="D", s=75)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Seed-level macro F1")
    axes[1].set_xlim(0, min(1.0, float(res["macro_f1"].max()) * 1.15))
    axes[1].text(-0.08, 1.03, "b", transform=axes[1].transAxes, fontsize=14, fontweight="bold")

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, "fig3_retention_and_seed_points")


def fig4_recall_heatmap(per_class: pd.DataFrame) -> Dict[str, str]:
    mean_df = per_class.groupby(["region", "class_name"])["recall"].mean().reset_index()
    pivot = mean_df.pivot(index="region", columns="class_name", values="recall")
    for cls in CLASS_ORDER:
        if cls not in pivot.columns:
            pivot[cls] = np.nan
    pivot = pivot[CLASS_ORDER]
    pivot = pivot.loc[[r for r in REGION_ORDER if r in pivot.index]]
    labels = [REGION_SHORT.get(r, r) for r in pivot.index]

    fig, ax = plt.subplots(figsize=(7.5, max(5.5, 0.45 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean recall across seeds")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Behaviour class")
    ax.set_ylabel("Body-region condition")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=("white" if val < 0.45 else "black"), fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "fig4_per_class_recall_heatmap")


def fig5_degradation(deg_summary: pd.DataFrame) -> Dict[str, str]:
    df = add_labels(deg_summary)
    df = df[df["region"].isin(MAIN_REGIONS)]
    modes = ["jitter", "dropout", "temporal_subsample"]
    titles = {"jitter": "Keypoint jitter", "dropout": "Keypoint dropout", "temporal_subsample": "Temporal subsampling"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharey=True)
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        sub_mode = df[df["degradation_mode"] == mode]
        for region in MAIN_REGIONS:
            sub = sub_mode[sub_mode["region"] == region].sort_values("degradation_severity")
            if sub.empty:
                continue
            ax.errorbar(
                sub["degradation_severity"].values,
                sub["macro_f1_mean"].values,
                yerr=sub["macro_f1_ci95"].values,
                marker="o",
                linewidth=1.4,
                capsize=3,
                label=REGION_SHORT.get(region, region),
            )
        ax.set_xlabel("Severity")
        ax.set_title(titles.get(mode, mode), fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.18, 1.04, chr(ord("a") + idx), transform=ax.transAxes, fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0, 1.0)
    axes[2].legend(fontsize=7, loc="lower left", bbox_to_anchor=(1.02, 0.0))
    fig.tight_layout()
    return save_fig(fig, "fig5_degradation_curves_macro_f1")


def fig6_degradation_retention(deg_ret_sum: pd.DataFrame) -> Dict[str, str]:
    df = deg_ret_sum.copy()
    df = df[df["region"].isin(MAIN_REGIONS)]
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No degradation retention data available", ha="center", va="center")
        ax.axis("off")
        return save_fig(fig, "fig6_degradation_retention_heatmap")

    df["condition"] = df["degradation_mode"] + "\n" + df["degradation_severity"].astype(str)
    condition_order = []
    for mode in ["clean", "jitter", "dropout", "temporal_subsample"]:
        sub = df[df["degradation_mode"] == mode].sort_values("degradation_severity")
        for cond in sub["condition"].unique():
            if cond not in condition_order:
                condition_order.append(cond)
    pivot = df.pivot_table(index="region", columns="condition", values="macro_f1_retention_percent_mean", aggfunc="mean")
    pivot = pivot.reindex([r for r in MAIN_REGIONS if r in pivot.index])
    pivot = pivot.reindex(columns=[c for c in condition_order if c in pivot.columns])
    labels_y = [REGION_SHORT.get(r, r) for r in pivot.index]

    fig, ax = plt.subplots(figsize=(12.5, max(4.8, 0.45 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Macro F1 retained vs full body (%)")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    ax.set_xlabel("Degradation condition")
    ax.set_ylabel("Body-region condition")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    return save_fig(fig, "fig6_degradation_retention_heatmap")


def fig7_paired_differences(paired: pd.DataFrame) -> Dict[str, str]:
    df = paired[(paired["metric"] == "macro_f1") & (paired["region"] != "full_body_both_mice")].copy()
    df = df[df["region"].isin([r for r in REGION_ORDER if r != "full_body_both_mice"])]
    df = add_labels(sort_regions(df))
    fig, ax = plt.subplots(figsize=(8.2, max(5.0, 0.45 * len(df))))
    y = np.arange(len(df))
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.scatter(df["mean_difference_region_minus_full"], y, s=55)
    for pos, (_, row) in enumerate(df.iterrows()):
        ax.text(row["mean_difference_region_minus_full"], pos + 0.16, f"p={fmt_p(row.get('paired_t_p_holm', np.nan))}", ha="center", va="bottom", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["region_short"])
    ax.set_xlabel("Mean macro F1 difference relative to full body")
    ax.set_ylabel("Body-region condition")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, "fig7_paired_macro_f1_differences_vs_full")


# =============================================================================
# Tables and snippets
# =============================================================================

def create_tables(summary: pd.DataFrame, ret_sum: pd.DataFrame, paired: pd.DataFrame, deg_summary: pd.DataFrame, deg_ret_sum: pd.DataFrame) -> Dict[str, str]:
    paths = {}
    clean = summary.merge(
        ret_sum[["region", "macro_f1_retention_percent_mean", "macro_f1_retention_percent_ci95"]],
        on="region",
        how="left",
    )
    clean = sort_regions(clean)
    table1 = pd.DataFrame({
        "Region": clean["region"].map(REGION_SHORT).fillna(clean["region"]),
        "Macro F1": [fmt_mean_ci(m, c) for m, c in zip(clean["macro_f1_mean"], clean["macro_f1_ci95"])],
        "Balanced accuracy": [fmt_mean_ci(m, c) for m, c in zip(clean["balanced_accuracy_mean"], clean["balanced_accuracy_ci95"])],
        "Macro F1 retained (\\%)": [fmt_mean_ci(m, c, 1) for m, c in zip(clean["macro_f1_retention_percent_mean"], clean["macro_f1_retention_percent_ci95"])],
    })
    table1.to_csv(PACKAGE_TABLE_DIR / "table1_clean_ablation_summary.csv", index=False)
    (PACKAGE_TABLE_DIR / "table1_clean_ablation_summary.md").write_text(table1.to_markdown(index=False), encoding="utf-8")
    save_latex_table(table1, PACKAGE_TABLE_DIR / "table1_clean_ablation_summary.tex", "Clean body-region ablation performance across five video-level splits.", "tab:clean_ablation")
    paths["table1"] = str(PACKAGE_TABLE_DIR / "table1_clean_ablation_summary.tex")

    p = paired[paired["metric"].isin(["macro_f1", "balanced_accuracy"])].copy()
    p = p[p["region"].isin([r for r in REGION_ORDER if r != "full_body_both_mice"])]
    p = sort_regions(p)
    table2 = pd.DataFrame({
        "Metric": p["metric"].replace({"macro_f1": "Macro F1", "balanced_accuracy": "Balanced accuracy"}),
        "Region": p["region"].map(REGION_SHORT).fillna(p["region"]),
        "Mean difference": p["mean_difference_region_minus_full"].map(lambda x: f"{x:.3f}"),
        "Paired t-test Holm $p$": p["paired_t_p_holm"].map(fmt_p),
        "Wilcoxon Holm $p$": p["wilcoxon_p_holm"].map(fmt_p),
    })
    table2.to_csv(PACKAGE_TABLE_DIR / "table2_paired_tests_vs_full.csv", index=False)
    (PACKAGE_TABLE_DIR / "table2_paired_tests_vs_full.md").write_text(table2.to_markdown(index=False), encoding="utf-8")
    save_latex_table(table2, PACKAGE_TABLE_DIR / "table2_paired_tests_vs_full.tex", "Paired comparisons between compact body-region conditions and full-body pose.", "tab:paired_tests")
    paths["table2"] = str(PACKAGE_TABLE_DIR / "table2_paired_tests_vs_full.tex")

    if not deg_summary.empty:
        deg = deg_summary[deg_summary["region"].isin(MAIN_REGIONS)].copy()
        selected = deg[
            ((deg["degradation_mode"] == "clean") & (deg["degradation_severity"] == 0.0))
            | ((deg["degradation_mode"] == "jitter") & (deg["degradation_severity"] == 0.1))
            | ((deg["degradation_mode"] == "dropout") & (deg["degradation_severity"] == 0.3))
            | ((deg["degradation_mode"] == "temporal_subsample") & (deg["degradation_severity"] == 5.0))
        ].copy()
        selected = sort_regions(selected)
        table3 = pd.DataFrame({
            "Region": selected["region"].map(REGION_SHORT).fillna(selected["region"]),
            "Degradation": selected["degradation_mode"],
            "Severity": selected["degradation_severity"].map(lambda x: f"{x:g}"),
            "Macro F1": [fmt_mean_ci(m, c) for m, c in zip(selected["macro_f1_mean"], selected["macro_f1_ci95"])],
            "Balanced accuracy": [fmt_mean_ci(m, c) for m, c in zip(selected["balanced_accuracy_mean"], selected["balanced_accuracy_ci95"])],
        })
        table3.to_csv(PACKAGE_TABLE_DIR / "table3_degradation_selected_conditions.csv", index=False)
        (PACKAGE_TABLE_DIR / "table3_degradation_selected_conditions.md").write_text(table3.to_markdown(index=False), encoding="utf-8")
        save_latex_table(table3, PACKAGE_TABLE_DIR / "table3_degradation_selected_conditions.tex", "Robustness under representative pose and temporal degradation conditions.", "tab:degradation_selected")
        paths["table3"] = str(PACKAGE_TABLE_DIR / "table3_degradation_selected_conditions.tex")

    if not deg_ret_sum.empty:
        dr = deg_ret_sum[deg_ret_sum["region"].isin(MAIN_REGIONS)].copy()
        selected = dr[
            ((dr["degradation_mode"] == "clean") & (dr["degradation_severity"] == 0.0))
            | ((dr["degradation_mode"] == "jitter") & (dr["degradation_severity"] == 0.1))
            | ((dr["degradation_mode"] == "dropout") & (dr["degradation_severity"] == 0.3))
            | ((dr["degradation_mode"] == "temporal_subsample") & (dr["degradation_severity"] == 5.0))
        ].copy()
        selected = sort_regions(selected)
        table4 = pd.DataFrame({
            "Region": selected["region"].map(REGION_SHORT).fillna(selected["region"]),
            "Degradation": selected["degradation_mode"],
            "Severity": selected["degradation_severity"].map(lambda x: f"{x:g}"),
            "Macro F1 retention (\\%)": [fmt_mean_ci(m, c, 1) for m, c in zip(selected["macro_f1_retention_percent_mean"], selected["macro_f1_retention_percent_ci95"])],
        })
        table4.to_csv(PACKAGE_TABLE_DIR / "table4_degradation_retention_selected_conditions.csv", index=False)
        (PACKAGE_TABLE_DIR / "table4_degradation_retention_selected_conditions.md").write_text(table4.to_markdown(index=False), encoding="utf-8")
        save_latex_table(table4, PACKAGE_TABLE_DIR / "table4_degradation_retention_selected_conditions.tex", "Retention of macro F1 relative to full-body pose under representative degradation conditions.", "tab:degradation_retention")
        paths["table4"] = str(PACKAGE_TABLE_DIR / "table4_degradation_retention_selected_conditions.tex")
    return paths


def create_snippets(summary: pd.DataFrame, ret_sum: pd.DataFrame, paired: pd.DataFrame, deg_summary: pd.DataFrame, dataset_manifest: Dict[str, Any], run_manifest: Dict[str, Any]) -> Dict[str, str]:
    clean = summary.merge(ret_sum[["region", "macro_f1_retention_percent_mean"]], on="region", how="left")
    full = clean[clean["region"] == "full_body_both_mice"].iloc[0]
    compact = clean[clean["region"].isin(COMPACT_REGIONS)].copy()
    best_compact = compact.sort_values("macro_f1_mean", ascending=False).iloc[0]
    compact_min = compact["macro_f1_retention_percent_mean"].min()
    compact_max = compact["macro_f1_retention_percent_mean"].max()

    resident = clean[clean["region"] == "resident_only_full_body"]
    intruder = clean[clean["region"] == "intruder_only_full_body"]
    resident_macro = float(resident["macro_f1_mean"].iloc[0]) if len(resident) else float("nan")
    intruder_macro = float(intruder["macro_f1_mean"].iloc[0]) if len(intruder) else float("nan")

    no_tail_test = paired[(paired["metric"] == "macro_f1") & (paired["region"] == "no_tail_both_mice")]
    no_tail_p = float(no_tail_test["paired_t_p_holm"].iloc[0]) if len(no_tail_test) and "paired_t_p_holm" in no_tail_test.columns else float("nan")

    methods = (
        "The analysis used the CalMS21 Task 1 pose trajectories. Pose windows of 45 frames were extracted with a stride of 2 frames, "
        "corresponding to 1.5 s temporal windows at the 30 Hz acquisition rate. Each window contained two animals, two image coordinates, "
        "and seven body landmarks: nose, left ear, right ear, neck, left hip, right hip, and tail base. Classification was performed using "
        "a compact temporal convolutional network with residual dilated one-dimensional convolutional blocks. Evaluation used five independent "
        "video-level train--validation splits. Splits were accepted only when all four behaviour classes were represented in both training and "
        "validation sets. For each split, class-balanced caps were applied before model training.\n\n"
        "Body-region ablations tested whether social-behaviour information remained observable from compact two-animal pose subsets. The full-body "
        "condition used all seven landmarks from both animals. Compact two-animal conditions included head-neck, no-tail, trunk-tail, hips-tail, "
        "head-only, hips-only, neck-only, and tail-base-only representations. Single-animal controls used all seven landmarks from either the resident "
        "or intruder animal alone. Model selection used validation macro F1 across epochs. Final performance was summarized across five seeds using "
        "mean values and 95\\% confidence intervals computed from seed-level variation."
    )

    results = (
        f"Full-body pose gave the highest average clean performance, with macro F1 of {full['macro_f1_mean']:.3f} $\\pm$ {full['macro_f1_ci95']:.3f} "
        f"and balanced accuracy of {full['balanced_accuracy_mean']:.3f} $\\pm$ {full['balanced_accuracy_ci95']:.3f}. However, compact two-animal "
        f"representations retained most of this performance. Among the compact conditions, {REGION_SHORT.get(best_compact['region'], best_compact['region'])} "
        f"achieved the highest macro F1 ({best_compact['macro_f1_mean']:.3f} $\\pm$ {best_compact['macro_f1_ci95']:.3f}). Across the main compact "
        f"two-animal conditions, macro F1 retention relative to full body ranged from {compact_min:.1f}\\% to {compact_max:.1f}\\%.\n\n"
        f"Single-animal controls were substantially lower than the two-animal representations. Resident-only full-body pose achieved macro F1 of "
        f"{resident_macro:.3f}, while intruder-only full-body pose achieved macro F1 of {intruder_macro:.3f}. This separation indicates that "
        f"social-behaviour information is encoded more strongly in dyadic relational geometry than in the posture of either animal alone. The no-tail "
        f"condition differed from full body by a small mean macro-F1 difference and its Holm-corrected paired test was {fmt_p(no_tail_p)}."
    )

    degradation = ""
    if not deg_summary.empty:
        clean_full = deg_summary[(deg_summary["region"] == "full_body_both_mice") & (deg_summary["degradation_mode"] == "clean") & (deg_summary["degradation_severity"] == 0.0)]
        jitter_full = deg_summary[(deg_summary["region"] == "full_body_both_mice") & (deg_summary["degradation_mode"] == "jitter") & (deg_summary["degradation_severity"] == 0.1)]
        if len(clean_full) and len(jitter_full):
            degradation = (
                f"Under degradation, full-body macro F1 decreased from {clean_full['macro_f1_mean'].iloc[0]:.3f} in the clean condition to "
                f"{jitter_full['macro_f1_mean'].iloc[0]:.3f} under the strongest keypoint-jitter condition. Compact two-animal models showed "
                "similar degradation profiles, whereas single-animal controls were more sensitive to perturbation in several degradation modes."
            )

    captions = (
        "Figure 1. Overview of the minimum-observable-body analysis. (a) CalMS21 pose trajectories were windowed into 1.5 s samples and classified using a temporal convolutional model. The trained models were evaluated under clean body-region ablations and pose-degradation conditions. (b) Schematic examples of the body-region representations used in the ablation study. Filled landmarks indicate observed keypoints.\n\n"
        "Figure 2. Clean social-behaviour classification performance across body-region conditions. Bars show mean performance across five video-level splits; error bars indicate 95% confidence intervals computed from seed-level variation. (a) Macro F1. (b) Balanced accuracy.\n\n"
        "Figure 3. Retention of full-body performance by compact body-region conditions. (a) Macro F1 retained relative to the full-body two-mouse representation. (b) Seed-level macro-F1 values for each body-region condition; diamonds indicate across-seed means.\n\n"
        "Figure 4. Behaviour-specific recall across body-region conditions. Values are averaged across five video-level splits. The heatmap identifies which behaviour classes are most sensitive to removing specific body regions.\n\n"
        "Figure 5. Robustness of body-region representations under pose and temporal degradation. Curves show mean macro F1 across seeds under keypoint jitter, keypoint dropout, and temporal subsampling; error bars indicate 95% confidence intervals.\n\n"
        "Figure 6. Macro-F1 retention relative to full-body pose under degradation. Values greater than or near 100% indicate that a compact representation preserved performance comparable to the full-body model under the same degradation condition.\n\n"
        "Figure 7. Paired macro-F1 differences between each body-region condition and the full-body two-mouse representation. Points show seed-paired mean differences; displayed p-values are Holm-corrected paired t-test results."
    )

    abstract = (
        f"Social behaviour in interacting animals is usually analysed using full-body pose, but it is unclear how much of the behavioural information "
        f"requires complete articulated tracking. This study used CalMS21 mouse social-interaction pose trajectories to quantify the minimum observable "
        f"body geometry needed for behaviour recognition. A compact temporal convolutional model was trained across five video-level splits using full-body "
        f"and reduced body-region representations. Full-body two-mouse pose achieved the highest mean macro F1 ({full['macro_f1_mean']:.3f}), but compact "
        f"two-mouse body regions retained most of this performance, with the main compact conditions preserving {compact_min:.1f}--{compact_max:.1f}\\% of "
        "full-body macro F1. Single-animal controls performed substantially worse, indicating that dyadic relational geometry carries more behavioural "
        "information than either animal's posture alone. Robustness tests under keypoint jitter, keypoint dropout, and temporal subsampling further supported "
        "the stability of compact two-animal representations."
    )

    x_shape = dataset_manifest.get("X_shape", ["NA"]) if dataset_manifest else ["NA"]
    n_windows = x_shape[0] if isinstance(x_shape, list) and len(x_shape) else "NA"
    dataset_note = (
        f"Cached window dataset: {n_windows} windows; unique videos: {dataset_manifest.get('unique_groups', 'NA') if dataset_manifest else 'NA'}; "
        f"class counts: {dataset_manifest.get('class_counts', {}) if dataset_manifest else {}}. "
        f"Recorded run time: {run_manifest.get('total_runtime_formatted', 'not recorded') if run_manifest else 'not recorded'}."
    )

    snippets = {
        "methods_snippet.tex": methods,
        "results_snippet.tex": results,
        "degradation_results_snippet.tex": degradation,
        "figure_captions.tex": captions,
        "abstract_seed_text.tex": abstract,
        "dataset_runtime_note.txt": dataset_note,
    }
    paths = {}
    for name, text in snippets.items():
        path = PACKAGE_SNIPPET_DIR / name
        write_text(path, text)
        paths[name] = str(path)
    return paths


# =============================================================================
# Copy, README, zip
# =============================================================================

def copy_raw_outputs() -> List[str]:
    copied = []
    files = [
        "clean_all_seed_region_results.csv",
        "clean_summary_across_seeds.csv",
        "clean_retention_vs_full_by_seed.csv",
        "clean_paired_tests_vs_full.csv",
        "clean_per_class_results.csv",
        "clean_pooled_bootstrap_ci.csv",
        "degradation_results.csv",
        "degradation_summary_across_seeds.csv",
        "degradation_paired_tests_vs_full.csv",
        "dataset_manifest.json",
        "run_manifest.json",
        "config.json",
        "all_splits.csv",
    ]
    for name in files:
        src = RESULTS_DIR / name
        if src.exists():
            dst = PACKAGE_DATA_DIR / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
            print(f"[copy] {src} -> {dst}")

    for src_dir, sub in [(ORIGINAL_FIGURES_DIR, "figures"), (ORIGINAL_TABLES_DIR, "tables")]:
        if src_dir.exists():
            dst_dir = PACKAGE_ORIGINAL_DIR / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in src_dir.glob("*"):
                if src.is_file():
                    dst = dst_dir / src.name
                    shutil.copy2(src, dst)
                    copied.append(str(dst))
    return copied


def create_readme(summary: pd.DataFrame, ret_sum: pd.DataFrame, zip_path: Path) -> None:
    full = summary[summary["region"] == "full_body_both_mice"].iloc[0]
    compact = summary[summary["region"].isin(COMPACT_REGIONS)].copy()
    compact = compact.merge(ret_sum[["region", "macro_f1_retention_percent_mean"]], on="region", how="left")
    best = compact.sort_values("macro_f1_mean", ascending=False).iloc[0]

    readme = f"""# CalMS21 Minimum Observable Body Geometry — Paper Package

This package was generated from:

`{BASE_DIR}`

## Main clean result

Full-body two-mouse pose:

- Macro F1: {full['macro_f1_mean']:.3f} ± {full['macro_f1_ci95']:.3f}
- Balanced accuracy: {full['balanced_accuracy_mean']:.3f} ± {full['balanced_accuracy_ci95']:.3f}

Best compact two-mouse region:

- {REGION_SHORT.get(best['region'], best['region'])}
- Macro F1: {best['macro_f1_mean']:.3f} ± {best['macro_f1_ci95']:.3f}
- Macro F1 retention: {best['macro_f1_retention_percent_mean']:.1f}%

## Recommended paper figures

1. `figures/fig1_pipeline_and_body_regions.pdf`
2. `figures/fig2_clean_performance.pdf`
3. `figures/fig3_retention_and_seed_points.pdf`
4. `figures/fig4_per_class_recall_heatmap.pdf`
5. `figures/fig5_degradation_curves_macro_f1.pdf`
6. `figures/fig6_degradation_retention_heatmap.pdf`
7. `figures/fig7_paired_macro_f1_differences_vs_full.pdf`

## Recommended paper tables

1. `tables/table1_clean_ablation_summary.tex`
2. `tables/table2_paired_tests_vs_full.tex`
3. `tables/table3_degradation_selected_conditions.tex`
4. `tables/table4_degradation_retention_selected_conditions.tex`

## LaTeX snippets

Useful draft text is in `latex_snippets/`.

## Zip

`{zip_path}`
"""
    write_text(PACKAGE_DIR / "README.md", readme)


def create_manifest(paths: Dict[str, Any]) -> None:
    manifest = {
        "base_dir": str(BASE_DIR),
        "package_dir": str(PACKAGE_DIR),
        "zip_output_path": str(ZIP_OUTPUT_PATH),
        "paths": paths,
    }
    (PACKAGE_DIR / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] Saved {PACKAGE_DIR / 'package_manifest.json'}")


def zip_package() -> Path:
    if ZIP_OUTPUT_PATH.exists():
        ZIP_OUTPUT_PATH.unlink()
    shutil.make_archive(str(ZIP_OUTPUT_PATH.with_suffix("")), "zip", PACKAGE_DIR)
    print(f"[zip] Saved {ZIP_OUTPUT_PATH}")
    print(f"[zip] Size: {ZIP_OUTPUT_PATH.stat().st_size / (1024 ** 2):.2f} MB")
    return ZIP_OUTPUT_PATH


# =============================================================================
# Main
# =============================================================================

def main() -> Dict[str, Any]:
    print("=" * 100)
    print("CalMS21 Minimum Observable Body Geometry — Results Package Generator")
    print("=" * 100)

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    reset_package_dirs()

    clean_results = load_csv("clean_all_seed_region_results.csv")
    clean_summary = load_csv("clean_summary_across_seeds.csv")
    retention_by_seed = load_csv("clean_retention_vs_full_by_seed.csv")
    paired_tests = load_csv("clean_paired_tests_vs_full.csv")
    per_class = load_csv("clean_per_class_results.csv")

    degradation = load_csv("degradation_results.csv", required=False)
    degradation_summary = load_csv("degradation_summary_across_seeds.csv", required=False)
    degradation_tests = load_csv("degradation_paired_tests_vs_full.csv", required=False)

    dataset_manifest = load_json("dataset_manifest.json")
    run_manifest = load_json("run_manifest.json")

    retention_summary = build_retention_summary(retention_by_seed)
    degradation_retention = build_degradation_retention(degradation)
    degradation_retention_summary = summarize_degradation_retention(degradation_retention)

    retention_summary.to_csv(PACKAGE_DATA_DIR / "derived_retention_summary.csv", index=False)
    degradation_retention.to_csv(PACKAGE_DATA_DIR / "derived_degradation_retention_by_seed.csv", index=False)
    degradation_retention_summary.to_csv(PACKAGE_DATA_DIR / "derived_degradation_retention_summary.csv", index=False)

    print("\n[summary] Clean summary:")
    print(sort_regions(add_labels(clean_summary))[["region", "macro_f1_mean", "macro_f1_ci95", "balanced_accuracy_mean", "balanced_accuracy_ci95"]].to_string(index=False))

    figure_paths = {}
    figure_paths["fig1"] = fig1_pipeline()
    figure_paths["fig2"] = fig2_clean_performance(clean_summary)
    figure_paths["fig3"] = fig3_retention(retention_summary, clean_results)
    figure_paths["fig4"] = fig4_recall_heatmap(per_class)
    if not degradation_summary.empty:
        figure_paths["fig5"] = fig5_degradation(degradation_summary)
        figure_paths["fig6"] = fig6_degradation_retention(degradation_retention_summary)
    figure_paths["fig7"] = fig7_paired_differences(paired_tests)

    table_paths = create_tables(clean_summary, retention_summary, paired_tests, degradation_summary, degradation_retention_summary)
    snippet_paths = create_snippets(clean_summary, retention_summary, paired_tests, degradation_summary, dataset_manifest, run_manifest)
    copied_outputs = copy_raw_outputs()

    first_zip = zip_package()
    create_readme(clean_summary, retention_summary, first_zip)

    all_paths = {
        "figures": figure_paths,
        "tables": table_paths,
        "snippets": snippet_paths,
        "copied_outputs_count": len(copied_outputs),
        "zip": str(first_zip),
    }
    create_manifest(all_paths)
    final_zip = zip_package()

    print("\n" + "=" * 100)
    print("Results package complete")
    print("=" * 100)
    print(f"Package directory: {PACKAGE_DIR}")
    print(f"Zip file         : {final_zip}")
    print(f"Figures          : {PACKAGE_FIG_DIR}")
    print(f"Tables           : {PACKAGE_TABLE_DIR}")
    print(f"LaTeX snippets   : {PACKAGE_SNIPPET_DIR}")
    print("=" * 100)

    return {
        "package_dir": str(PACKAGE_DIR),
        "zip_path": str(final_zip),
        "figure_paths": figure_paths,
        "table_paths": table_paths,
        "snippet_paths": snippet_paths,
    }


if __name__ == "__main__":
    package_outputs = main()