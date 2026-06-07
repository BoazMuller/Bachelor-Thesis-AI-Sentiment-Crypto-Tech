
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("results/figures/")
OUT.mkdir(exist_ok=True)

def add_box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#222222", lw=1.5,
            fontsize=11, weight="normal", color="#111111", radius=0.03):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=color, wrap=True, zorder=4)
    return patch

def add_arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.8, rad=0.0, zorder=2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=16, linewidth=lw,
        color=color, shrinkA=8, shrinkB=8,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder
    )
    ax.add_patch(arr)
    return arr

def add_line_arrow(ax, points, color="#333333", lw=1.8, zorder=1):
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                          linewidth=lw, color=color, shrinkA=0, shrinkB=8, zorder=zorder)
    ax.add_patch(arr)
    return arr

def make_data_flow():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    blue = "#E9F2FF"; blue_edge = "#2F6DB3"
    green = "#EAF7E7"; green_edge = "#3F8F46"
    orange = "#FFF2DD"; orange_edge = "#D98C20"
    purple = "#F2EAFF"; purple_edge = "#7B4CC2"
    red = "#FDECEA"; red_edge = "#D93025"
    gray = "#F3F4F6"; dark = "#111827"

    ax.text(0.035, 0.955, "Empirical workflow: data streams into the models",
            fontsize=24, weight="bold", ha="left", va="top", color=dark)
    ax.text(0.035, 0.915, "Clean presentation version: sentiment construction, expectation filtering, and connectedness modelling",
            fontsize=13, ha="left", va="top", color="#4B5563")

    # Lanes
    ax.text(0.035, 0.775, "AI SENTIMENT STREAM", fontsize=12, weight="bold", color=blue_edge)
    ax.text(0.035, 0.365, "MARKET VOLATILITY STREAM", fontsize=12, weight="bold", color=orange_edge)

    # Top lane: sentiment
    add_box(ax, 0.04, 0.64, 0.16, 0.105, "AI text\nReddit posts\nGDELT headlines",
            blue, blue_edge, fontsize=10.8, weight="bold")
    add_box(ax, 0.24, 0.64, 0.16, 0.105, "RoBERTa\nsentiment scores\npositive − negative",
            blue, blue_edge, fontsize=10.3)
    add_box(ax, 0.44, 0.64, 0.16, 0.105, "Daily aggregation\n+ dynamic factor model\nRaw AIS",
            blue, blue_edge, fontsize=10.3, weight="bold")
    add_box(ax, 0.64, 0.64, 0.16, 0.105, "Expectation filter\nprediction markets\n+ controls",
            green, green_edge, fontsize=10.3, weight="bold")
    add_box(ax, 0.84, 0.64, 0.13, 0.105, "EAIS\nspeculative AI\nsentiment",
            red, red_edge, lw=2.2, fontsize=10.5, weight="bold", color="#B42318")

    # Bottom lane: market/model stream
    add_box(ax, 0.04, 0.24, 0.16, 0.105, "Financial assets\nBTC, NDX\nNVDA, GOOGL, MSFT",
            orange, orange_edge, fontsize=10.8, weight="bold")
    add_box(ax, 0.24, 0.24, 0.16, 0.105, "Log returns\ntrading-day alignment",
            orange, orange_edge, fontsize=10.3)
    add_box(ax, 0.44, 0.24, 0.16, 0.105, "Univariate\nEGARCH(1,1)\nconditional volatilities",
            orange, orange_edge, fontsize=10.3, weight="bold")
    add_box(ax, 0.64, 0.24, 0.16, 0.105, "TVP-VAR\nconnectedness\nTCI, TO, FROM",
            orange, orange_edge, fontsize=10.3, weight="bold")
    add_box(ax, 0.84, 0.24, 0.13, 0.105, "Spillover\nregressions\non EAIS + controls",
            gray, "#6B7280", fontsize=10.2, weight="bold")

    # Direct effects model between lanes
    add_box(ax, 0.39, 0.445, 0.22, 0.095, "ARMA-EGARCH\nEAIS tests direct return and volatility effects",
            "#FFF7ED", orange_edge, fontsize=10.5, weight="bold")

    # Output callouts
    add_box(ax, 0.06, 0.075, 0.37, 0.08, "Direct-effects finding:\nEAIS predicts next-day Bitcoin returns; not NASDAQ-100 returns or volatility.",
            "#F8FAFC", "#CBD5E1", fontsize=10.5)
    add_box(ax, 0.57, 0.075, 0.37, 0.08, "Connectedness finding:\nEAIS strengthens firm-level BTC–AI-equity spillovers; BTC becomes more of a receiver.",
            "#F8FAFC", "#CBD5E1", fontsize=10.5)

    # Clean horizontal arrows
    for y in [0.6925, 0.2925]:
        add_arrow(ax, 0.20, y, 0.24, y, "#4B5563", lw=1.8)
        add_arrow(ax, 0.40, y, 0.44, y, "#4B5563", lw=1.8)
        add_arrow(ax, 0.60, y, 0.64, y, "#4B5563", lw=1.8)
        add_arrow(ax, 0.80, y, 0.84, y, "#4B5563", lw=1.8)

    # EAIS to models/regressions
    add_line_arrow(ax, [(0.905, 0.64), (0.905, 0.54), (0.61, 0.54)],
                   red_edge, lw=1.8, zorder=1)  # EAIS to ARMA-EGARCH
    add_line_arrow(ax, [(0.905, 0.64), (0.905, 0.40), (0.905, 0.345)],
                   red_edge, lw=1.8, zorder=1)  # EAIS down to spillover regressions

    # Financial returns to ARMA-EGARCH
    add_arrow(ax, 0.32, 0.345, 0.43, 0.445, orange_edge, lw=1.7, rad=0.08)

    ax.text(0.04, 0.018,
            "How to present it: the thesis first builds an expectation-adjusted AI sentiment index, then uses it in direct asset models and in connectedness regressions.",
            fontsize=10.5, color="#4B5563", ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(OUT / "figure_1_data_flow_model.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure_1_data_flow_model.svg", bbox_inches="tight")
    plt.close(fig)

def make_contribution_matrix():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    green = "#EAF7E7"; green_edge = "#3F8F46"
    orange = "#FFF7ED"; orange_edge = "#D97706"
    red = "#FDECEA"; red_edge = "#D93025"
    dark = "#111827"

    ax.text(0.04, 0.955, "Where this thesis contributes",
            fontsize=24, weight="bold", ha="left", va="top", color=dark)
    ax.text(0.04, 0.915, "A 2×2 positioning matrix for the academic design",
            fontsize=13, ha="left", va="top", color="#4B5563")

    left = 0.21; bottom = 0.16; gap = 0.012
    cell_w = 0.345; cell_h = 0.285; row_w = 0.145; header_h = 0.085

    ax.text(left + cell_w + gap/2, bottom + 2*cell_h + gap + header_h + 0.035,
            "Empirical focus", fontsize=13, weight="bold", ha="center", color="#374151")
    ax.text(0.078, bottom + cell_h + gap/2, "Sentiment design", fontsize=13,
            weight="bold", rotation=90, ha="center", va="center", color="#374151")

    add_box(ax, left, bottom + 2*cell_h + gap, cell_w, header_h,
            "Individual asset effects\nreturns / conditional volatility",
            "#F8FAFC", "#111827", lw=1.2, fontsize=12, weight="bold")
    add_box(ax, left + cell_w + gap, bottom + 2*cell_h + gap, cell_w, header_h,
            "Cross-market transmission\nvolatility connectedness",
            "#F8FAFC", "#111827", lw=1.2, fontsize=12, weight="bold")
    add_box(ax, left - row_w - gap, bottom + cell_h + gap, row_w, cell_h,
            "Aggregate or raw\nsentiment", "#F8FAFC", "#111827", lw=1.2, fontsize=12, weight="bold")
    add_box(ax, left - row_w - gap, bottom, row_w, cell_h,
            "Expectation-adjusted\nAI sentiment", "#F8FAFC", "#111827", lw=1.2, fontsize=12, weight="bold")

    add_box(ax, left, bottom + cell_h + gap, cell_w, cell_h,
            "Known literature\n\nSentiment explains returns and risk\nin crypto and equity markets.\n\nOften studied inside one asset class\nor one broad market.",
            green, green_edge, lw=1.3, fontsize=11)
    add_box(ax, left + cell_w + gap, bottom + cell_h + gap, cell_w, cell_h,
            "Related literature\n\nSentiment and attention are linked\nto dynamic spillovers.\n\nBut often uses broad sentiment,\nnot AI-specific narrative sentiment.",
            green, green_edge, lw=1.3, fontsize=11)
    add_box(ax, left, bottom, cell_w, cell_h,
            "Methodological step\n\nRaw AI sentiment is filtered through\nprediction markets and macro controls.\n\nThis separates observable expectations\nfrom residual narrative sentiment.",
            orange, orange_edge, lw=1.3, fontsize=11)

    main_x = left + cell_w + gap; main_y = bottom
    add_box(ax, main_x, main_y, cell_w, cell_h,
            "MAIN CONTRIBUTION\n\nSpeculative AI sentiment as a\ncross-market transmission channel.\n\nEAIS → connectedness regressions\nfor BTC–NDX and BTC–AI-equity systems.",
            red, red_edge, lw=2.6, fontsize=11.5, weight="bold", color="#B42318")
    outer = FancyBboxPatch((main_x - 0.010, main_y - 0.010), cell_w + 0.020, cell_h + 0.020,
                           boxstyle="round,pad=0.012,rounding_size=0.035",
                           facecolor="none", edgecolor=red_edge, linewidth=2.2,
                           linestyle=(0, (4, 4)), zorder=5)
    ax.add_patch(outer)

    callout = "“I do not only test whether AI sentiment moves assets; I test whether its speculative component changes cross-market volatility transmission.”"
    ax.text(0.50, 0.075, callout, fontsize=12, color="#374151", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.2", fc="#F8FAFC", ec="#CBD5E1", lw=1.2))

    fig.tight_layout()
    fig.savefig(OUT / "figure_2_contribution_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure_2_contribution_matrix.svg", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    make_data_flow()
    make_contribution_matrix()
    print("Saved figures to", OUT)
