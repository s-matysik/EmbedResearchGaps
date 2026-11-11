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
    """Component diagram: what each stage consumes, produces and hands on.

    Rewritten after review: the first version was a flow of undifferentiated
    rounded boxes.  This one separates the three stages visually, names the
    module behind every component, marks the two modes as alternatives rather
    than a sequence, and numbers the processing order so the diagram can be
    read without the surrounding text.
    """
    fig, ax = plt.subplots(figsize=(7.1, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    band = {"front": "#eef3f8", "modes": "#fdf1e7", "back": "#edf6f1"}
    edge = {"front": "#2a6099", "modes": "#c45a1a", "back": "#1b7a52"}

    def stage(y, height, key, title):
        ax.add_patch(FancyBboxPatch(
            (1.5, y), 97, height, boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=0, facecolor=band[key], zorder=0))
        ax.text(3.0, y + height - 3.4, title, fontsize=TICK, style="italic",
                color=edge[key], zorder=2)

    def component(x, y, w, h, title, module, detail, key, fontsize=TICK):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.9",
            linewidth=1.1, edgecolor=edge[key], facecolor="white", zorder=2))
        ax.text(x + w / 2, y + h - 3.6, title, ha="center", fontsize=fontsize,
                weight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 7.4, module, ha="center", fontsize=fontsize - 1.2,
                family="monospace", color=edge[key], zorder=3)
        if detail:
            ax.text(x + w / 2, y + 2.2, detail, ha="center", va="bottom",
                    fontsize=fontsize - 1.2, color="#333333", zorder=3)

    def arrow(x1, y1, x2, y2, label=None, style="-|>"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
            linewidth=1.0, color="#444444", zorder=4,
            connectionstyle="arc3,rad=0.0", shrinkA=1, shrinkB=1))
        if label:
            ax.text((x1 + x2) / 2 + 1.2, (y1 + y2) / 2, label, fontsize=TICK - 2,
                    color="#444444", va="center", zorder=4)

    stage(70, 28, "front", "1  Shared front end")
    component(4, 74, 27, 18, "Load and normalise", "corpus / text",
              "Scopus, WoS, CSV\nauthor keywords only", "front")
    component(36.5, 74, 27, 18, "Embed", "encoders",
              "7 backends\noffline default", "front")
    component(69, 74, 27, 18, "Select analysis corpus", "clustering",
              "top N nearest to the\nproblem statement", "front")
    arrow(31, 83, 36.5, 83)
    arrow(63.5, 83, 69, 83)

    stage(36, 31, "modes", "2  Two alternative modes")
    component(4, 40, 44, 22, "Mode A  keywords first", "pipelines.keywords_first",
              "cluster keywords, flag those beyond\nthe 95th distance percentile of their cluster",
              "modes")
    component(52, 40, 44, 22, "Mode B  articles first", "pipelines.articles_first",
              "cluster articles, then mine each cluster's keyword\n"
              "sub-space: emerging / cross-cluster / combination",
              "modes")
    arrow(50, 70, 26, 62)
    arrow(50, 70, 74, 62)
    ax.text(50, 66.5, "either or both", fontsize=TICK - 2, ha="center", color="#444444")

    stage(2, 32, "back", "3  Shared back end")
    component(4, 8, 28, 22, "Stability", "consensus",
              "repeat across seeds\nand corpus sizes;\nper-candidate support", "back")
    component(36, 8, 28, 22, "Assessment", "validation",
              "5-model panel,\ndisjoint control,\nblinded expert sheets", "back")
    component(68, 8, 28, 22, "Outputs", "results / viz",
              "candidate tables,\nrun configuration,\nfigures, diagnostics", "back")
    # Both modes return the same result object, so both feed every back-end
    # component; draw the convergence explicitly rather than implying that
    # each mode owns a different downstream box.
    ax.plot([26, 26, 74, 74], [40, 35, 35, 40], color="#444444", linewidth=1.0, zorder=4)
    for target in (18, 50, 82):
        arrow(50, 35, target, 30)
    arrow(32, 19, 36, 19)
    arrow(64, 19, 68, 19)

    ax.text(50, 4.0, "external rankings enter at stage 3 through "
                     "external.load_external_ranking",
            ha="center", fontsize=TICK - 1.5, style="italic", color="#444444")
    fig.tight_layout(pad=0.3)
    return save(fig, "fig1_architecture.png")


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
    # Outlines and the count come from the deposited candidate table, not from
    # this render's run: the figure must mark the candidate set the article
    # reports. Re-embedding the corpus through a hosted endpoint can shift the
    # partition slightly, which is exactly the instability Section 3.2 measures,
    # but a figure that disagreed with Table 1 would be a defect rather than a
    # finding.
    deposited = pd.read_csv("results/management/articles_first_gaps.csv")
    gaps = deposited.set_index("keyword")
    gaps = gaps[~gaps.index.duplicated(keep="first")]
    live = set(result.gaps["keyword"])
    print(f"  deposited candidates {gaps.index.nunique()} | this run {len(live)} | "
          f"shared {len(live & set(gaps.index))}", flush=True)

    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    for cluster in sorted(result.cluster_labels):
        idx = [i for i, kw in enumerate(keywords) if int(lookup.get(kw, -1)) == cluster]
        if not idx:
            continue
        # Area grows with the square root of document frequency: the linear
        # scaling of the first version made a handful of frequent keywords
        # dominate the panel and obscure their neighbours.
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            s=[10 + 16 * np.sqrt(float(frequency.get(keywords[i], 1))) for i in idx],
            color=PALETTE[cluster % len(PALETTE)], alpha=0.75,
            linewidths=0.4, edgecolors="white",
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
    # The earlier title asserted that candidates sit at the edge of every
    # cluster.  Mode B candidates are not defined by keyword-space periphery,
    # and t-SNE does not preserve global distance, so the claim is replaced by
    # the statistic that can be measured: where candidate distances from their
    # own sub-centroid fall in the distribution of all keyword distances.
    distances = result.keywords.groupby("keyword")["distance_from_subcentroid"].max()
    ranks = distances.rank(pct=True) * 100
    candidate_percentile = float(ranks[ranks.index.isin(gaps.index)].median())
    ax.set_title(
        f"Author-keyword space of the management corpus: t-SNE of {len(keywords)} "
        f"keyword embeddings\n{gaps.index.nunique()} Mode B candidates outlined; their "
        f"median distance from the sub-cluster centroid falls at the "
        f"{candidate_percentile:.0f}th percentile of all keywords"
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

    # The unit of analysis is the keyword, matching Section 3.3: the five
    # provider judgements of one keyword are aggregated before the arms are
    # compared.  Reading the judgement-level summary here, as an earlier
    # version did, annotated the figure with p-values from a different design.
    stats_path = Path("results/keyword_level_comparisons.json")
    keyword_level = json.loads(stats_path.read_text())
    deltas: list[dict[str, object]] = []
    for name, pretty in corpora.items():
        for mode, label in modes.items():
            record = keyword_level[f"{name}|{mode}"]
            deltas.append({
                "corpus": pretty, "mode": label,
                "delta": record["gap_delta_pp"],
                "ci": record["gap_ci_pp"],
                "p": record["gap_perm_p"],
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
        errors = np.array([[value - float(d["ci"][0]) for value, d in zip(values, subset)],
                           [float(d["ci"][1]) - value for value, d in zip(values, subset)]])
        ax.bar(positions + offset, values, width, color=colour, edgecolor="black",
               linewidth=0.4, label=pretty_mode)
        ax.errorbar(positions + offset, values, yerr=errors, fmt="none",
                    ecolor="black", elinewidth=0.9, capsize=3)
        for x, value, d in zip(positions + offset, values, subset):
            top = float(d["ci"][1])
            ax.text(x, top + 1.4, f"p={float(d['p']):.2f}", ha="center",
                    fontsize=TICK - 1, color=GREY)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, list(corpora.values()), fontsize=TICK)
    ax.set_ylabel("GAP rate of candidates minus control (pp)")
    ax.set_ylim(-75, 55)
    ax.set_title("Candidate minus control GAP rate,\nkeyword level, with 95 % CI")
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
    ax.set_title("Mean Novelty Score at three stages\nof the pipeline (SE bars)")
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
                # Choose the annotation colour from the luminance of the cell
                # it sits on, not from the raw value: the colour map is
                # rescaled per panel, so a fixed value threshold put white
                # text on pale cells.
                rgba = image.cmap(image.norm(values[i, j]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        fontsize=TICK - 1,
                        color="white" if luminance < 0.5 else "black")
        ax.set_title(pretty)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).set_label(
            "cosine similarity", fontsize=TICK
        )
    fig.suptitle("Pairwise cosine similarity of the six article-cluster centroids",
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
