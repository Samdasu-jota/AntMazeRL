"""
Phase 3.3 visualization — evaluation_results.json -> comparison figures.

Re-plots from JSON only (no re-evaluation). 4 panels + standalone hero (factor impact).
  (A) Per-factor impact tornado  — how many pp each intervention added (★ key)
  (B) Baselines vs RL (short maze) — random / BC / RL pre-flip / RL post-flip
  (C) Flip-fix gamechanger         — success up + flip rate down (same env)
  (D) Map difficulty & full-map transfer — short 88 / full zero-shot 39 / full fine-tune 73

All chart text is English. Captions live in a single figure footnote (never overlap bars).

Usage: python -m scripts.plot_comparison [--json outputs/evaluation_results.json]
"""
import os
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

POS, NEG = "#2e8b57", "#c0392b"        # positive = green, negative = red
BLUE, GRAY, GOLD = "#2c6fbb", "#9aa0a6", "#e0a93b"

# key -> (English label, environment tag) for the tornado
FACTOR_EN = {
    "F1_learning":       ("Training (random → RL)",        "[short maze]"),
    "F2_bc_vs_scratch":  ("Imitation (BC) vs scratch",          "[plane 3m]"),
    "F3_geometry":       ("Geometry: full → short +A*",     "[maze, honest]"),
    "F4_flipfix":        ("★ Flip-fix (upright)",           "[short maze, same env]"),
    "F5_map_difficulty": ("Map difficulty: short → full",   "[upright policy]"),
    "F6_full_finetune":  ("Full-map fine-tune",                  "[full maze]"),
}
ENTRY_EN = {
    "random_short": "random",
    "bc_short": "BC\n(imitation)",
    "rl_preflip_short": "RL pre-flip\n(inverted)",
    "rl_postflip_short": "RL post-flip\n(upright)",
    "rl_upright_full_zeroshot": "Full maze\nzero-shot",
    "rl_full_finetune": "Full maze\nfine-tune",
}


def panel_tornado(ax, fd):
    labels = [f"{FACTOR_EN[d['key']][0]}\n{FACTOR_EN[d['key']][1]}" for d in fd]
    deltas = [d["delta"] for d in fd]
    colors = [POS if x >= 0 else NEG for x in deltas]
    y = np.arange(len(fd))
    ax.barh(y, deltas, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    span = max(abs(min(deltas)), abs(max(deltas)))
    for i, d in enumerate(deltas):
        ax.text(d + (2.5 if d >= 0 else -2.5), i, f"{d:+.0f}pp",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=9.5, fontweight="bold", color=colors[i])
    ax.set_xlim(-span - 22, span + 22)
    ax.set_xlabel("Δ Success rate (percentage points)")
    ax.set_title("(A) Per-factor impact on success rate  ★", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)


def panel_baselines(ax, E):
    keys = ["random_short", "bc_short", "rl_preflip_short", "rl_postflip_short"]
    vals = [E[k]["success_rate"] for k in keys]
    labels = ["random", "BC\n(imitation)", "RL pre-flip\n(inverted)", "RL post-flip\n(upright)"]
    colors = [GRAY, GRAY, GOLD, POS]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("(B) Baselines vs RL  —  short maze (honest r1.0)", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def panel_flipfix(ax, E):
    pre, post = E["rl_preflip_short"], E["rl_postflip_short"]
    x = np.arange(2)
    w = 0.36
    succ = [pre["success_rate"], post["success_rate"]]    # 58, 88
    flip = [pre["flip_rate"], post["flip_rate"]]          # 99, 11
    b1 = ax.bar(x - w / 2, succ, w, color=POS, label="Success rate ↑")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, flip, w, color=NEG, label="Flip rate ↓")
    ax.set_xticks(x)
    ax.set_xticklabels(["RL pre-flip\n(inverted)", "RL post-flip\n(upright)"])
    ax.set_ylabel("Success rate (%)", color=POS)
    ax2.set_ylabel("Flip rate (%)", color=NEG)
    ax.tick_params(axis="y", colors=POS)
    ax2.tick_params(axis="y", colors=NEG)
    ax.set_ylim(0, 112)
    ax2.set_ylim(0, 112)
    for bx, v in zip(b1, succ):
        ax.text(bx.get_x() + bx.get_width() / 2, v + 2, f"{v:.0f}%", ha="center",
                fontsize=10, fontweight="bold", color=POS)
    for bx, v in zip(b2, flip):
        ax2.text(bx.get_x() + bx.get_width() / 2, v + 2, f"{v:.0f}%", ha="center",
                 fontsize=10, fontweight="bold", color=NEG)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8,
              framealpha=0.9, ncol=2)
    ax.set_title("(C) ★ Flip-fix  —  same maze, posture only (58→88%)", fontweight="bold")


def panel_mapdiff(ax, E):
    keys = ["rl_postflip_short", "rl_upright_full_zeroshot", "rl_full_finetune"]
    vals = [E[k]["success_rate"] for k in keys]
    labels = ["Short maze\n(upright 88%)", "Full maze\nzero-shot", "Full maze\nfine-tune"]
    colors = [POS, NEG, BLUE]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("(D) Map difficulty & full-map transfer", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


FOOTNOTE = ("Deterministic eval, 100 episodes, seed0=20000.   "
            "Pre-flip-fix policies evaluated with posture-termination OFF (original regime), "
            "so 58 / 81 / 87% reproduce the log.   "
            "Panel D: 88→39% is a harder map (full 6m pillar), not a regression.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/evaluation_results.json")
    ap.add_argument("--out", default="outputs/images/comparison.png")
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))
    E, fd = data["entries"], data["factor_deltas"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    panel_tornado(axes[0, 0], fd)
    panel_baselines(axes[0, 1], E)
    panel_flipfix(axes[1, 0], E)
    panel_mapdiff(axes[1, 1], E)
    fig.suptitle("AntMazeRL — Per-factor performance comparison (random vs BC vs RL + ablation)",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.018, FOOTNOTE, ha="center", va="bottom", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=3.0, w_pad=3.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"saved: {args.out}")

    # standalone hero (factor impact)
    fig2, ax = plt.subplots(figsize=(10, 5.5))
    panel_tornado(ax, fd)
    fig2.tight_layout()
    hero = os.path.join(os.path.dirname(args.out), "factor_impact.png")
    fig2.savefig(hero, dpi=130)
    plt.close(fig2)
    print(f"saved: {hero}")


if __name__ == "__main__":
    main()
