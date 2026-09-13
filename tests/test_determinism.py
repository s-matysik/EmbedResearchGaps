"""Reproducibility across processes, not just within one.

A same-process repeat cannot detect order dependence on the interpreter's
hash seed, because both runs share it.  These tests run the pipelines in
subprocesses started with different ``PYTHONHASHSEED`` values and require the
candidate tables to be identical -- the property a published run depends on.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")
TESTS = str(Path(__file__).resolve().parent)

SCRIPT = """
import json, sys
sys.path[:0] = [%(src)r, %(tests)r]
from conftest import build_synthetic_frame
from embedresearchgaps import (
    ClusteringConfig, EncoderConfig, GapConfig, RunConfig,
    articles_first, from_dataframe, keywords_first,
)

corpus = from_dataframe(build_synthetic_frame())
encoder = EncoderConfig(name="tfidf-svd", dimensions=64, extra={"random_state": 42})
config = RunConfig(
    top_n_articles=60,
    article_encoder=encoder,
    keyword_encoder=encoder,
    clustering=ClusteringConfig(k_min=2, k_max=6),
    gaps=GapConfig(max_gaps_per_cluster=3),
)
out = {}
for name, runner in (("articles_first", articles_first), ("keywords_first", keywords_first)):
    result = runner(corpus, config)
    out[name] = {
        "keywords": list(result.gaps["keyword"]) if not result.gaps.empty else [],
        "scores": [round(float(s), 10) for s in result.gaps["score"]] if not result.gaps.empty else [],
        "k": result.n_clusters,
        "labels": {str(k): v for k, v in result.cluster_labels.items()},
        "keyword_space": list(result.keyword_index),
    }
print(json.dumps(out))
"""


def run_with_hash_seed(seed: str) -> dict:
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT % {"src": SRC, "tests": TESTS}],
        capture_output=True, text=True, env=environment, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def two_runs() -> tuple[dict, dict]:
    return run_with_hash_seed("0"), run_with_hash_seed("12345")


@pytest.mark.parametrize("mode", ["articles_first", "keywords_first"])
def test_candidate_table_is_hash_seed_invariant(two_runs, mode: str) -> None:
    first, second = two_runs
    assert first[mode]["keywords"] == second[mode]["keywords"]
    assert first[mode]["scores"] == second[mode]["scores"]


@pytest.mark.parametrize("mode", ["articles_first", "keywords_first"])
def test_clustering_is_hash_seed_invariant(two_runs, mode: str) -> None:
    first, second = two_runs
    assert first[mode]["k"] == second[mode]["k"]
    assert first[mode]["labels"] == second[mode]["labels"]


@pytest.mark.parametrize("mode", ["articles_first", "keywords_first"])
def test_keyword_space_row_order_is_stable(two_runs, mode: str) -> None:
    first, second = two_runs
    assert first[mode]["keyword_space"] == second[mode]["keyword_space"]
    assert first[mode]["keyword_space"], "keyword space must not be empty"


def test_keyword_space_order_follows_first_appearance(synthetic_corpus) -> None:
    from embedresearchgaps.encoders import get_encoder
    from embedresearchgaps.config import EncoderConfig
    from embedresearchgaps.keywords import build_keyword_space

    encoder = get_encoder(EncoderConfig(name="tfidf-svd", dimensions=32))
    space = build_keyword_space(synthetic_corpus.keyword_lists, encoder)
    expected: list[str] = []
    for keywords in synthetic_corpus.keyword_lists:
        for keyword in dict.fromkeys(keywords):
            if keyword not in expected:
                expected.append(keyword)
    assert space.keywords == expected


def test_document_frequency_order_follows_first_appearance() -> None:
    from embedresearchgaps.text import keyword_frequencies

    records = [["beta term", "alpha term"], ["alpha term"], ["gamma term", "beta term"]]
    counter = keyword_frequencies(records)
    assert list(counter) == ["beta term", "alpha term", "gamma term"]
    assert counter["alpha term"] == 2
