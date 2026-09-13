"""TF-IDF back-off stages and result serialisation edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from embedresearchgaps.text import TfidfSalience, compute_tfidf_salience
from embedresearchgaps.text import _best_vocabulary_match as best_match


class TestVocabularyBackOff:
    VOCABULARY = {
        "customer engagement": 0.30,
        "engagement": 0.20,
        "brand loyalty programme": 0.10,
        "digital": 0.05,
    }

    def test_exact_stage(self) -> None:
        assert best_match("customer engagement", self.VOCABULARY) == (0.30, "exact")

    def test_contains_stage_picks_the_strongest_superstring(self) -> None:
        score, stage = best_match("engagemen", self.VOCABULARY)
        assert stage == "contains"
        assert score == 0.30

    def test_overlap_stage_requires_half_the_tokens(self) -> None:
        score, stage = best_match("loyalty programme", self.VOCABULARY)
        assert stage in {"contains", "overlap"}
        assert score == 0.10

    def test_fallback_stage(self) -> None:
        score, stage = best_match("zzz qqq", self.VOCABULARY)
        assert (score, stage) == (TfidfSalience.FALLBACK, "fallback")

    def test_single_token_miss_falls_back(self) -> None:
        assert best_match("qqq", self.VOCABULARY)[1] == "fallback"

    def test_every_stage_appears_in_a_realistic_corpus(self) -> None:
        salience = compute_tfidf_salience(
            [
                ["customer engagement", "brand loyalty programme"],
                ["customer engagement", "digital"],
                ["engagement quality"],
            ]
        )
        stages = set(salience.match_statistics())
        assert "exact" in stages


class TestResultSerialisation:
    def test_numpy_scalars_survive_json(self, synthetic_corpus, offline_config) -> None:
        import json

        from embedresearchgaps import articles_first

        result = articles_first(synthetic_corpus, offline_config)
        result.diagnostics["numpy_int"] = np.int64(3)
        result.diagnostics["numpy_float"] = np.float64(0.5)
        result.diagnostics["numpy_array"] = np.arange(3)
        result.diagnostics["a_set"] = {"a", "b"}
        payload = json.loads(result.to_json())
        assert payload["diagnostics"]["numpy_int"] == 3
        assert payload["diagnostics"]["numpy_float"] == 0.5
        assert payload["diagnostics"]["numpy_array"] == [0, 1, 2]
        assert sorted(payload["diagnostics"]["a_set"]) == ["a", "b"]

    def test_gap_type_counts_of_an_empty_run(self, synthetic_corpus, offline_config) -> None:
        import dataclasses

        from embedresearchgaps import GapConfig, articles_first

        config = dataclasses.replace(
            offline_config, gaps=GapConfig(min_tfidf=99.0, max_keyword_frequency=0)
        )
        result = articles_first(synthetic_corpus, config)
        assert result.gap_type_counts() == {}
        assert result.top_gaps(5).empty

    def test_run_config_round_trips_to_json(self, offline_config) -> None:
        import json

        payload = json.loads(offline_config.to_json())
        assert payload["gaps"]["emerging_weights"]["tfidf"] == 0.30
        assert isinstance(payload["gaps"]["methodological_tools"], list)
        assert payload["gaps"]["combination_similarity_range"] == [0.25, 0.65]
