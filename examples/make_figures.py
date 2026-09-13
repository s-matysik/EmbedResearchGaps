"""Build the manuscript figures (SoftwareX allows at most six).

Fig. 1  software architecture and the two operating modes
Fig. 2  management case: author-keyword space with candidates outlined
Fig. 3  finance case: candidates per type and top-ranked candidates
Fig. 4  inter-cluster centroid similarity, both case studies
Fig. 5  stability: candidate-set overlap across seeds and corpus sizes
Fig. 6  inter-cluster thematic overlap, both case studies
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

sys.path.insert(0, "embedresearchgaps/src")
sys.path.append(os.getcwd())

FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

BASE, SMALL, TICK = 9, 8, 7
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
GREY = "#4D4D4D"

plt.rcParams.update({
    "font.size": BASE,
    "axes.titlesize": BASE,
    "axes.labelsize": BASE,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "legend.fontsize": SMALL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlelocation": "left",
})

CASES = {"management": "Management", "finance": "Economics & finance"}


def save(fig: plt.Figure, name: str) -> str:
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Fig. 1 -- architecture
# --------------------------------------------------------------------------- #
def figure_architecture() -> str:
    fig, ax = plt.subplots(figsize=(7.1, 5.2))
    ax.set_xlim(-2, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")

    def box(x, y, w, h, text, colour, fontsize=SMALL, weight="normal"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                linewidth=0.9, edgecolor=colour, facecolor=colour + "1A",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color="black", weight=weight, linespacing=1.35)

    def arrow(x1, y1, x2, y2, colour=GREY, style="-|>"):
        ax.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            mutation_scale=9, linewidth=0.9, color=colour,
                            shrinkA=1, shrinkB=1)
        )

    # shared front end
    box(1, 57, 22, 11, "Bibliographic export\n(Scopus / WoS / CSV)", GREY)
    box(27, 57, 20, 11, "load_csv\nnormalise keywords", GREY)
    box(51, 57, 18, 11, "encoders\n7 backends", PALETTE[4])
    box(73, 57, 25, 11, "Semantic ranking to the\nproblem statement\n-> top-N corpus", PALETTE[4])
    arrow(23, 62.5, 27, 62.5)
    arrow(47, 62.5, 51, 62.5)
    arrow(69, 62.5, 73, 62.5)

    # mode A
    box(1, 33, 44, 13,
        "Mode A   keywords_first\ncluster the author keywords,\nflag the periphery of each keyword cluster",
        PALETTE[0], fontsize=SMALL)
    box(1, 18, 44, 9, "Peripheral Keyword\ndistance from centroid > cluster percentile",
        PALETTE[0])
    arrow(23, 33, 23, 27)

    # mode B
    box(54, 33, 44, 13,
        "Mode B   articles_first\ncluster the articles, label the themes,\nmine a keyword sub-space in each cluster",
        PALETTE[1], fontsize=SMALL)
    box(54, 18, 13, 9, "Emerging\nConcept", PALETTE[1])
    box(69, 18, 13, 9, "Conceptual\nCombination", PALETTE[1])
    box(84, 18, 14, 9, "Cross-Cluster\nConcept", PALETTE[1])
    arrow(76, 33, 60.5, 27)
    arrow(76, 33, 75.5, 27)
    arrow(76, 33, 91, 27)

    arrow(80, 57, 76, 46)
    arrow(80, 57, 23, 46)

    # shared back end
    box(1, 1, 29, 10, "consensus_gaps\nsupport across seeds", PALETTE[2])
    box(35, 1, 29, 10, "validation\n5-model LLM panel,\nexpert sheets", PALETTE[3])
    box(69, 1, 29, 10, "viz + results.save\nfigures, tables,\ndiagnostics", PALETTE[5])
    arrow(23, 18, 15, 11)
    arrow(23, 18, 49, 11)
    arrow(76, 18, 50, 11)
    arrow(76, 18, 83, 11)

    ax.text(-1, 70, "Shared front end", fontsize=SMALL, color=GREY, style="italic")
    ax.text(-1, 48.5, "Two operating modes", fontsize=SMALL, color=GREY, style="italic")
    ax.text(-1, 13.5, "Shared back end", fontsize=SMALL, color=GREY, style="italic")
    return save(fig, "fig1_architecture.png")


# --------------------------------------------------------------------------- #
# Fig. 2 -- keyword space of the management case
# --------------------------------------------------------------------------- #
def figure_keyword_space() -> str:
    from embedresearchgaps import articles_first, load_csv
    from embedresearchgaps.viz import embed_2d
    from run_case import PROBLEMS, build_config, embed_corpus

    name = "management"
    corpus = load_csv(f"corpora/{name}.csv")
    vectors = embed_corpus(name, corpus, PROBLEMS[name])
    result = articles_first(corpus, build_config(PROBLEMS[name], vectors, 50))

    coords = embed_2d(result.embeddings["keywords"], 42)
    keywords = result.keyword_index
    lookup = result.keywords.groupby("keyword")["article_cluster"].agg(
        lambda s: s.value_counts().idxmax()
    )
    frequency = result.keywords.groupby("keyword")["frequency"].max()
    gaps = result.gaps.set_index("keyword")

    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    for cluster in sorted(result.cluster_labels):
        idx = [i for i, kw in enumerate(keywords) if int(lookup.get(kw, -1)) == cluster]
        if not idx:
            continue
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            s=[14 + 9 * float(frequency.get(keywords[i], 1)) for i in idx],
            color=PALETTE[cluster % len(PALETTE)], alpha=0.55,
            linewidths=0.3, edgecolors="white",
            label=f"{result.cluster_labels[cluster]}",
        )
    # Outline every candidate, but only label the highest-scoring ones, and
    # drop a label whose anchor sits closer than `spacing` to one already
    # placed: three labels stacked on one dense corner are unreadable, and a
    # missing label costs less than an illegible one.
    top = gaps.nlargest(6, "score")
    span = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])))
    spacing = 0.16 * span
    placed: list[np.ndarray] = []
    for i, keyword in enumerate(keywords):
        if keyword not in gaps.index:
            continue
        point = coords[i]
        ax.scatter(point[0], point[1], s=95, facecolors="none",
                   edgecolors="black", linewidths=1.0, zorder=3)
        if keyword not in top.index:
            continue
        if any(np.linalg.norm(point - other) < spacing for other in placed):
            continue
        placed.append(point)
        ax.annotate(
            gaps.loc[keyword, "keyword_display"] if "keyword_display" in gaps.columns
            else keyword,
            (point[0], point[1]), fontsize=TICK,
            xytext=(8, 6), textcoords="offset points", zorder=4,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY,
                            shrinkA=0, shrinkB=2),
        )
    ax.scatter([], [], s=95, facecolors="none", edgecolors="black",
               linewidths=1.0, label="candidate gap")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.10)
    ax.set_title(
        "Candidate gaps sit at the edge of every keyword cluster\n"
        f"{len(keywords)} author keywords, {len(gaps)} candidates, "
        f"{result.n_clusters} article clusters (management case)"
    )
    ax.annotate("", xy=(0.055, 0.055), xytext=(0.055, 0.15), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<-", lw=0.7, color=GREY))
    ax.annotate("", xy=(0.055, 0.055), xytext=(0.155, 0.055), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<-", lw=0.7, color=GREY))
    ax.text(0.165, 0.05, "t-SNE 1", transform=ax.transAxes, fontsize=TICK, color=GREY)
    ax.text(0.05, 0.16, "t-SNE 2", transform=ax.transAxes, fontsize=TICK, color=GREY)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=TICK, markerscale=0.9, handletextpad=0.5)
    return save(fig, "fig2_keyword_space.png")


# --------------------------------------------------------------------------- #
# Fig. 3 -- candidates of the finance case
# --------------------------------------------------------------------------- #
def figure_finance_gaps() -> str:
    gaps = pd.read_csv("results/finance/articles_first_gaps.csv")
    order = ["Emerging Concept", "Conceptual Combination", "Cross-Cluster Concept"]
    colours = {name: PALETTE[i] for i, name in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), width_ratios=[1, 1.7])

    counts = [int((gaps["gap_type"] == name).sum()) for name in order]
    axes[0].bar(range(3), counts, color=[colours[n] for n in order],
                edgecolor="black", linewidth=0.4, width=0.62)
    for i, value in enumerate(counts):
        axes[0].text(i, value + 0.4, str(value), ha="center", fontsize=SMALL)
    axes[0].set_xticks(range(3), ["Emerging\nconcept", "Conceptual\ncombination",
                                  "Cross-cluster\nconcept"], fontsize=TICK)
    axes[0].set_ylabel("candidate gaps")
    axes[0].set_ylim(0, max(counts) * 1.18)
    axes[0].set_title("Rare single-study concepts dominate")

    top = gaps.nlargest(12, "score").iloc[::-1]
    axes[1].barh(range(len(top)), top["score"],
                 color=[colours[t] for t in top["gap_type"]],
                 edgecolor="black", linewidth=0.4, height=0.68)
    axes[1].set_yticks(range(len(top)), [str(k)[:38] for k in top["keyword_display"]],
                       fontsize=TICK)
    axes[1].set_xlabel("composite score (higher = stronger candidate)")
    axes[1].set_xlim(0, 0.86)
    axes[1].set_title("Twelve highest-ranked candidates")
    # No legend: the left panel's category axis already names each type in its
    # own colour, and both panels share the palette.
    fig.tight_layout()
    return save(fig, "fig3_finance_gaps.png")


# --------------------------------------------------------------------------- #
# Fig. 4 -- LLM panel validation
# --------------------------------------------------------------------------- #
def figure_validation() -> str:
    """Panel A: GAP-rate shift against a control set disjoint from the
    candidates, with the permutation p-value of the Novelty Score test.

    Panel B: the decomposition of mean Novelty Score into what the lexical and
    frequency filters contribute (keywords failing a filter -> eligible pool)
    and what the embedding-based ranking adds on top of them (eligible pool ->
    selected candidates).
    """
    modes = {"articles_first": "Mode B", "keywords_first": "Mode A"}
    corpora = {"management": "Management", "finance": "Economics\n& finance",
               "gamification": "Gamification"}

    deltas: list[dict[str, object]] = []
    for name, pretty in corpora.items():
        payload = json.loads(
            Path(f"results/{name}/validation_summary_disjoint.json").read_text()
        )
        for mode, label in modes.items():
            comparison = payload[mode]["comparison"]
            deltas.append({
                "corpus": pretty, "mode": label,
                "delta": comparison["gap_rate_delta_pp"],
                "p": comparison["novelty_test"]["permutation_p"],
            })

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), width_ratios=[1.2, 1])

    ax = axes[0]
    positions = np.arange(len(corpora))
    width = 0.36
    for offset, (mode, label) in zip((-width / 2, width / 2), modes.items()):
        subset = [d for d in deltas if d["mode"] == label]
        colour = PALETTE[1] if label == "Mode B" else PALETTE[0]
        values = [float(d["delta"]) for d in subset]
        pretty_mode = ("Mode B (articles first)" if label == "Mode B"
                       else "Mode A (keywords first)")
        ax.bar(positions + offset, values, width, color=colour, edgecolor="black",
               linewidth=0.4, label=pretty_mode)
        for x, d in zip(positions + offset, subset):
            value = float(d["delta"])
            ax.text(x, value + (0.8 if value >= 0 else -1.9),
                    f"p={float(d['p']):.2f}", ha="center", fontsize=TICK - 1, color=GREY)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, list(corpora.values()), fontsize=TICK)
    ax.set_ylabel("GAP rate of candidates minus control (pp)")
    ax.set_ylim(-15, 18)
    ax.set_title("No reliable enrichment over the\nnon-candidate keyword population")
    ax.legend(frameon=False, fontsize=TICK - 1, loc="upper left")

    ax = axes[1]
    arms = ["NOT_ELIGIBLE", "ELIGIBLE", "SELECTED"]
    arm_labels = ["fails a\nfilter", "passes filters,\nnot selected", "selected by\nthe ranking"]
    for index, (name, pretty) in enumerate([("management", "Management"),
                                            ("finance", "Economics & finance")]):
        payload = json.loads(
            Path(f"results/{name}/matched_arm_summary_disjoint.json").read_text()
        )
        filters = payload["eligible_vs_not_eligible"]
        ranking = payload["selected_vs_eligible"]
        means = [filters["all_novelty"]["mean"],
                 filters["selected_novelty"]["mean"],
                 ranking["selected_novelty"]["mean"]]
        errors = [
            filters["all_novelty"]["std"] / np.sqrt(filters["all_novelty"]["n"]),
            filters["selected_novelty"]["std"] / np.sqrt(filters["selected_novelty"]["n"]),
            ranking["selected_novelty"]["std"] / np.sqrt(ranking["selected_novelty"]["n"]),
        ]
        ax.errorbar(np.arange(3) + (index - 0.5) * 0.08, means, yerr=errors,
                    fmt="o-", capsize=3, markersize=5, linewidth=1.2,
                    color=PALETTE[index * 2], label=pretty)
    ax.set_xticks(range(3), arm_labels, fontsize=TICK - 1)
    ax.set_ylabel("mean Novelty Score (1-10)")
    ax.set_title("Filters raise novelty on both corpora;\nthe ranking then diverges")
    ax.legend(frameon=False, fontsize=TICK, loc="lower right")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return save(fig, "fig6_validation.png")


# --------------------------------------------------------------------------- #
# Fig. 5 -- stability
# --------------------------------------------------------------------------- #
def figure_stability() -> str:
    sizes = pd.read_csv("results/sensitivity/corpus_size_sensitivity.csv")
    seeds = pd.read_csv("results/sensitivity/seed_stability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))

    modes = {"articles_first": ("Mode B (articles first)", PALETTE[1]),
             "keywords_first": ("Mode A (keywords first)", PALETTE[0])}
    for mode, (pretty, colour) in modes.items():
        subset = sizes[
            (sizes["mode"] == mode)
            & (sizes["corpus"].isin(CASES))
            & (sizes["n_articles"] != 50)          # N = 50 is the reference itself
        ]
        grouped = subset.groupby("n_articles")["jaccard_vs_n50"].mean()
        axes[0].plot(grouped.index, grouped.to_numpy(), "o-", color=colour,
                     linewidth=1.3, markersize=5, label=pretty)
    axes[0].axvline(50, color=GREY, linestyle=":", linewidth=0.9)
    axes[0].text(51, 0.92, "reference\nN = 50", fontsize=TICK - 1, color=GREY)
    axes[0].set_xlabel("analysis-corpus size N")
    axes[0].set_ylabel("Jaccard overlap with the N = 50 list")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Candidate sets barely overlap\nacross corpus sizes")
    axes[0].legend(frameon=False, fontsize=TICK, loc="upper right")
    axes[0].grid(alpha=0.2)

    subset = seeds[seeds["corpus"].isin(CASES)]
    positions = np.arange(len(subset))
    colours = [modes[m][1] for m in subset["mode"]]
    axes[1].barh(positions, subset["mean_pairwise_jaccard"], color=colours,
                 edgecolor="black", linewidth=0.4, height=0.6)
    axes[1].errorbar(
        subset["mean_pairwise_jaccard"], positions,
        xerr=[subset["mean_pairwise_jaccard"] - subset["min_pairwise_jaccard"],
              subset["max_pairwise_jaccard"] - subset["mean_pairwise_jaccard"]],
        fmt="none", ecolor="black", capsize=2.5, linewidth=0.7,
    )
    axes[1].set_yticks(
        positions,
        [f"{CASES[c]}\n{'B' if m == 'articles_first' else 'A'}"
         for c, m in zip(subset["corpus"], subset["mode"])],
        fontsize=TICK,
    )
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("mean pairwise Jaccard across 5 k-means seeds")
    axes[1].set_title("and across initialisations\n(bars: min-max)")
    fig.tight_layout()
    return save(fig, "fig5_stability.png")


# --------------------------------------------------------------------------- #
# Fig. 6 -- inter-cluster overlap
# --------------------------------------------------------------------------- #
def figure_centroids() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4))
    for ax, (name, pretty) in zip(axes, CASES.items()):
        matrix = pd.read_csv(
            f"results/{name}/articles_first_centroid_similarity.csv", index_col=0
        )
        values = matrix.to_numpy()
        image = ax.imshow(values, cmap="YlOrRd", vmin=float(values.min()), vmax=1.0)
        labels = json.loads(
            Path(f"results/{name}/case_summary.json").read_text()
        )["articles_first"]["cluster_labels"]
        ticks = [f"C{i}" for i in range(len(values))]
        ax.set_xticks(range(len(values)), ticks, fontsize=TICK)
        ax.set_yticks(range(len(values)), ticks, fontsize=TICK)
        for i in range(len(values)):
            for j in range(len(values)):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        fontsize=TICK - 1,
                        color="white" if values[i, j] > 0.62 else "black")
        ax.set_title(pretty)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).set_label(
            "cosine similarity", fontsize=TICK
        )
    fig.suptitle("Article clusters overlap thematically but stay distinct",
                 fontsize=BASE, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig4_centroids.png")


BUILDERS = {
    "fig1": figure_architecture,
    "fig2": figure_keyword_space,
    "fig3": figure_finance_gaps,
    "fig4": figure_centroids,
    "fig5": figure_stability,
    "fig6": figure_validation,
}

if __name__ == "__main__":
    for key in sys.argv[1:] or list(BUILDERS):
        try:
            print(key, "->", BUILDERS[key](), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(key, "FAILED", type(exc).__name__, exc, flush=True)
