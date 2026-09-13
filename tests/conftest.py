"""Shared fixtures: a synthetic corpus with planted, known gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embedresearchgaps import ClusteringConfig, EncoderConfig, GapConfig, RunConfig, from_dataframe

# Three thematic groups.  Every group has core keywords that recur in most of
# its records, "established" keywords that recur a few times, and planted rare
# two-word keywords that the Type-1 detector is expected to surface.
GROUPS: dict[str, dict[str, object]] = {
    "supply": {
        "vocabulary": (
            "supply chain resilience disruption logistics supplier network "
            "inventory procurement sourcing risk mitigation capacity"
        ),
        "core": ["supply chain", "resilience"],
        "established": ["supplier selection", "inventory policy", "risk mitigation"],
        "planted_gaps": ["circular sourcing", "nearshoring strategy"],
    },
    "workforce": {
        "vocabulary": (
            "employee wellbeing engagement burnout remote work autonomy "
            "human resource retention turnover leadership motivation team"
        ),
        "core": ["employee engagement", "remote work"],
        "established": ["job autonomy", "turnover intention", "leadership style"],
        "planted_gaps": ["algorithmic management", "digital presenteeism"],
    },
    "finance": {
        "vocabulary": (
            "credit risk lending default portfolio bank capital liquidity "
            "interest rate borrower scoring financial institution"
        ),
        "core": ["credit risk", "bank lending"],
        "established": ["default probability", "capital buffer", "loan pricing"],
        "planted_gaps": ["climate transition", "embedded finance"],
    },
}

#: Keywords deliberately confined to one group, so that the Type-3 detector
#: sees them as present elsewhere but absent from the other clusters.
GROUP_EXCLUSIVE = {"supply": "tariff shock", "workforce": "skill mismatch", "finance": "shadow banking"}

#: Two frequent keywords of the *same* group that never appear together in one
#: record, which is what the Type-2 detector looks for (it works inside one
#: article cluster).  Guaranteed by never adding a second "established"
#: keyword to records of the finance group.
NEVER_COOCCURRING = ("default probability", "capital buffer")


def build_synthetic_frame(
    n_per_group: int = 20,
    seed: int = 7,
    include_methodological: bool = True,
) -> pd.DataFrame:
    """A bibliographic frame with three themes and planted gap keywords."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group_index, (group, spec) in enumerate(GROUPS.items()):
        words = str(spec["vocabulary"]).split()
        core = list(spec["core"])
        established = list(spec["established"])
        planted = list(spec["planted_gaps"])
        for i in range(n_per_group):
            title_words = rng.choice(words, size=6, replace=False)
            abstract_words = rng.choice(words, size=28, replace=True)
            keywords = [core[i % len(core)]]
            if i % 2 == 0:
                keywords.append(core[(i + 1) % len(core)])
            keywords.append(established[i % len(established)])
            if i % 5 == 0 and group != "finance":
                keywords.append(established[(i + 2) % len(established)])
            if i == 3:
                keywords.append(planted[0])
            if i == 11:
                keywords.append(planted[1])
            if i in (7, 13):
                keywords.append(GROUP_EXCLUSIVE[group])
            if include_methodological and i % 6 == 0:
                keywords.append("structural equation model")
            rows.append(
                {
                    "Title": f"{group.title()} study {i}: " + " ".join(title_words),
                    "Abstract": " ".join(abstract_words),
                    "Author Keywords": "; ".join(dict.fromkeys(keywords)),
                    "Year": int(2015 + (i % 10)),
                    "Cited by": int(rng.integers(0, 120)),
                    "DOI": f"10.0000/{group}.{i}",
                    "Authors": f"Author {group[0].upper()}{i}",
                    "Source title": f"Journal of {group.title()} Research",
                }
            )
    frame = pd.DataFrame(rows)
    first, second = NEVER_COOCCURRING
    together = frame["Author Keywords"].str.contains(first, regex=False) & frame[
        "Author Keywords"
    ].str.contains(second, regex=False)
    assert not together.any(), "the Type-2 target pair must never co-occur"
    return frame


def unit_vector(angle_degrees: float) -> list[float]:
    """A unit vector in the plane, embedded in R^3 with a zero third axis.

    Lets a test place keywords at exact cosine similarities: two vectors at
    angles ``a`` and ``b`` have ``cos(a - b)`` similarity.
    """
    radians = np.deg2rad(angle_degrees)
    return [float(np.cos(radians)), float(np.sin(radians)), 0.0]


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return build_synthetic_frame()


@pytest.fixture(scope="session")
def synthetic_corpus(synthetic_frame: pd.DataFrame):
    return from_dataframe(synthetic_frame)


@pytest.fixture()
def offline_config() -> RunConfig:
    """A deterministic, network-free configuration for the whole test suite."""
    encoder = EncoderConfig(name="tfidf-svd", dimensions=64, extra={"random_state": 42})
    return RunConfig(
        top_n_articles=60,
        article_encoder=encoder,
        keyword_encoder=encoder,
        clustering=ClusteringConfig(n_clusters=3, k_selection="fixed", random_state=42),
        gaps=GapConfig(max_gaps_per_cluster=3),
        random_state=42,
    )
