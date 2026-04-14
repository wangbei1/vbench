#!/usr/bin/env python3
"""Plot temporal quality curves for multiple methods.

Reads `scores_{T}s.json` files produced by run_temporal_curve_eval.sh and
generates a figure with one subplot per VBench dimension.

Usage:
    python3 plot_temporal_curve.py <output_root> [--out figure.pdf]

The output_root is expected to contain subdirectories per method, each with
scores_{5,15,30,45,60,80,100}s.json files.
"""
import argparse
import json
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

DIMS = [
    ("subject consistency",    "Subject Consistency"),
    ("background consistency", "Background Consistency"),
    ("motion smoothness",      "Motion Smoothness"),
    ("dynamic degree",         "Dynamic Degree"),
    ("aesthetic quality",      "Aesthetic Quality"),
    ("imaging quality",        "Imaging Quality"),
]

# Soft pastel palette
COLORS = {
    "ours":           "#E07A5F",  # warm coral
    "longlive":       "#81B29A",  # sage green
    "reward_forcing": "#3D5A80",  # dusty blue
}

# Friendly display names
DISPLAY_NAMES = {
    "ours":           "FocalReward (Ours)",
    "longlive":       "LongLive",
    "reward_forcing": "Reward Forcing",
}

MARKERS = {
    "ours":           "o",
    "longlive":       "s",
    "reward_forcing": "^",
}


def load_method_scores(method_dir):
    """Return {length_s: {dim_name: raw_score}}."""
    results = {}
    for fpath in sorted(glob(os.path.join(method_dir, "scores_*s.json"))):
        fname = os.path.basename(fpath)
        try:
            length = int(fname.replace("scores_", "").replace("s.json", ""))
        except ValueError:
            continue
        with open(fpath) as f:
            results[length] = json.load(f)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root", help="Folder with one subdir per method")
    ap.add_argument("--out", default="temporal_curve.pdf",
                    help="Output figure path (.pdf or .png)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Method names (subdirs) to plot, in order")
    args = ap.parse_args()

    if args.methods is None:
        methods = sorted(d for d in os.listdir(args.output_root)
                         if os.path.isdir(os.path.join(args.output_root, d)))
    else:
        methods = args.methods

    method_data = {}
    for m in methods:
        mdir = os.path.join(args.output_root, m)
        method_data[m] = load_method_scores(mdir)
        print(f"{m}: lengths = {sorted(method_data[m].keys())}")

    # Plot
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.edgecolor": "#444444",
    })

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()

    for i, (dim_key, dim_label) in enumerate(DIMS):
        ax = axes[i]
        for m in methods:
            data = method_data[m]
            if not data:
                continue
            lengths = sorted(data.keys())
            ys = []
            for L in lengths:
                v = data[L].get(dim_key)
                if v is None:
                    ys.append(np.nan)
                else:
                    ys.append(v * 100)  # scale to percentage
            color = COLORS.get(m, None)
            marker = MARKERS.get(m, "o")
            label = DISPLAY_NAMES.get(m, m)
            ax.plot(lengths, ys,
                    marker=marker, markersize=6,
                    linewidth=1.8, color=color,
                    label=label,
                    markeredgecolor="white", markeredgewidth=0.8)

        ax.set_title(dim_label, fontsize=12, pad=8, color="#222222")
        ax.set_xlabel("Video length (s)", fontsize=10, color="#444444")
        ax.set_ylabel("Score", fontsize=10, color="#444444")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_color("#888888")

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=len(methods),
               frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
