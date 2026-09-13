"""Annotation protocol, annotator panel, metrics and expert concordance."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from embedresearchgaps import GapConfig, articles_first
from embedresearchgaps.validation import (
    Annotation,
    permutation_mean_difference,
    selected_and_control_keywords,
    CallableAnnotator,
    ChatAnnotator,
    LABELS,
    annotations_to_frame,
    available_providers,
    build_expert_sheet,
    build_prompt,
    compare_keyword_sets,
    expert_llm_concordance,
    expert_report,
    fleiss_kappa,
    get_annotator,
    krippendorff_alpha_ordinal,
    label_rates,
    load_expert_sheet,
    mean_novelty,
    pairwise_agreement,
    parse_annotation_payload,
    per_keyword_consensus,
    selected_and_all_keywords,
    validate_gaps,
    write_expert_sheet,
)


def keywords_from_prompt(prompt: str) -> list[str]:
    """Recover the numbered keyword list a prompt was built from."""
    block = prompt.split("Keywords to evaluate:\n", 1)[1].split("\n\nReturn JSON")[0]
    return [re.sub(r"^\d+\.\s*", "", line).strip() for line in block.splitlines() if line.strip()]


def make_stub(
    name: str,
    gap_keywords: set[str],
    gap_novelty: float = 7.0,
    other_novelty: float = 2.0,
    omit: set[str] | None = None,
):
    """A deterministic annotator that labels ``gap_keywords`` as GAP."""
    omit = omit or set()

    def completion(prompt: str) -> str:
        entries = [
            {
                "keyword": keyword,
                "novelty": gap_novelty if keyword in gap_keywords else other_novelty,
                "label": "GAP" if keyword in gap_keywords else "EXPLORED",
                "rationale": "stub",
            }
            for keyword in keywords_from_prompt(prompt)
            if keyword not in omit
        ]
        return "```json\n" + json.dumps({"annotations": entries}) + "\n```"

    return CallableAnnotator(name, completion, batch_size=50)


class TestProtocol:
    def test_prompt_contains_problem_rubric_and_keywords(self) -> None:
        prompt = build_prompt(["alpha term", "beta term"], "How does X affect Y?", "management")
        assert "How does X affect Y?" in prompt
        assert "Novelty Score (1-10)" in prompt
        assert "1. alpha term" in prompt and "2. beta term" in prompt
        assert "management" in prompt

    def test_empty_keyword_list_raises(self) -> None:
        with pytest.raises(ValueError, match="no keywords"):
            build_prompt([], "problem")

    def test_parses_fenced_json(self) -> None:
        reply = '```json\n{"annotations": [{"keyword": "alpha term", "novelty": 8, "label": "GAP"}]}\n```'
        annotations = parse_annotation_payload(reply, ["alpha term"], "stub")
        assert annotations[0].label == "GAP"
        assert annotations[0].novelty == 8.0
        assert annotations[0].is_gap

    def test_matches_keywords_case_insensitively(self) -> None:
        reply = '{"annotations": [{"keyword": "Alpha Term", "novelty": 5, "label": "GAP"}]}'
        assert parse_annotation_payload(reply, ["alpha term"], "stub")[0].keyword == "alpha term"

    def test_unknown_keywords_and_labels_are_dropped(self) -> None:
        reply = json.dumps(
            {
                "annotations": [
                    {"keyword": "alpha term", "novelty": 5, "label": "GAP"},
                    {"keyword": "not asked", "novelty": 5, "label": "GAP"},
                    {"keyword": "beta term", "novelty": 5, "label": "MAYBE"},
                ]
            }
        )
        annotations = parse_annotation_payload(reply, ["alpha term", "beta term"], "stub")
        assert [a.keyword for a in annotations] == ["alpha term"]

    def test_malformed_reply_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            parse_annotation_payload("I cannot help with that.", ["alpha"], "stub")

    @pytest.mark.parametrize("novelty", [0, 11, -3])
    def test_novelty_outside_the_scale_is_rejected(self, novelty: int) -> None:
        with pytest.raises(ValueError, match="outside"):
            Annotation("alpha", "GAP", novelty, "stub")

    def test_invalid_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            Annotation("alpha", "PROBABLY", 5, "stub")

    def test_label_set_matches_the_published_protocol(self) -> None:
        assert LABELS == ("GAP", "EXPLORED", "LOW IMPORTANCE", "METHOD", "NOISE")


class TestAnnotators:
    def test_callable_annotator_batches(self) -> None:
        seen: list[int] = []

        def completion(prompt: str) -> str:
            batch = keywords_from_prompt(prompt)
            seen.append(len(batch))
            return json.dumps(
                {"annotations": [{"keyword": k, "novelty": 4, "label": "GAP"} for k in batch]}
            )

        annotator = CallableAnnotator("stub", completion, batch_size=2)
        annotations = annotator.annotate([f"kw{i} term" for i in range(5)], "problem")
        assert seen == [2, 2, 1]
        assert len(annotations) == 5

    def test_partial_reply_yields_fewer_annotations(self) -> None:
        annotator = make_stub("partial", {"alpha term"}, omit={"beta term"})
        annotations = annotator.annotate(["alpha term", "beta term"], "problem")
        assert [a.keyword for a in annotations] == ["alpha term"]

    def test_providers_are_registered(self) -> None:
        assert {"openai", "anthropic", "google", "deepseek", "moonshot", "xai"} <= set(
            available_providers()
        )

    def test_chat_annotator_reports_a_missing_key(self) -> None:
        annotator = get_annotator("openai", key_env="ERG_ABSENT_KEY")
        with pytest.raises(RuntimeError, match="ERG_ABSENT_KEY"):
            annotator.complete("prompt")

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown provider"):
            ChatAnnotator("llama-local")

    def test_default_temperature_is_zero(self) -> None:
        assert ChatAnnotator("openai").temperature == 0.0


class TestMetrics:
    @pytest.fixture()
    def frame(self) -> pd.DataFrame:
        rows = []
        for annotator, labels in {
            "a": ["GAP", "GAP", "EXPLORED", "METHOD"],
            "b": ["GAP", "EXPLORED", "EXPLORED", "METHOD"],
            "c": ["GAP", "GAP", "EXPLORED", "METHOD"],
        }.items():
            for keyword, label in zip(["k1", "k2", "k3", "k4"], labels):
                rows.append(
                    {
                        "annotator": annotator,
                        "keyword_set": "SELECTED",
                        "keyword": keyword,
                        "label": label,
                        "novelty": 7.0 if label == "GAP" else 2.0,
                        "rationale": "",
                    }
                )
        return pd.DataFrame(rows)

    def test_label_rates_sum_to_one(self, frame: pd.DataFrame) -> None:
        rates = label_rates(frame)
        assert sum(rates.values()) == pytest.approx(1.0)
        assert rates["GAP"] == pytest.approx(5 / 12)

    def test_label_rates_of_empty_frame(self) -> None:
        assert label_rates(pd.DataFrame()) == {label: 0.0 for label in LABELS}

    def test_mean_novelty_reports_dispersion(self, frame: pd.DataFrame) -> None:
        stats = mean_novelty(frame)
        assert stats["n"] == 12
        assert stats["mean"] == pytest.approx((5 * 7 + 7 * 2) / 12)
        assert stats["share_ge_5"] == pytest.approx(5 / 12)

    def test_consensus_flags_unanimity(self, frame: pd.DataFrame) -> None:
        consensus = per_keyword_consensus(frame)
        k1 = consensus[consensus["keyword"] == "k1"].iloc[0]
        k2 = consensus[consensus["keyword"] == "k2"].iloc[0]
        assert k1["unanimous"] and k1["gap_share"] == 1.0
        assert not k2["unanimous"] and k2["gap_votes"] == 2

    def test_fleiss_kappa_is_one_for_perfect_agreement(self) -> None:
        rows = [
            {"annotator": a, "keyword": k, "label": label, "novelty": 5.0, "keyword_set": "S", "rationale": ""}
            for k, label in (("k1", "GAP"), ("k2", "EXPLORED"))
            for a in ("a", "b", "c")
        ]
        assert fleiss_kappa(pd.DataFrame(rows)) == pytest.approx(1.0)

    def test_fleiss_kappa_is_undefined_for_a_single_label(self) -> None:
        rows = [
            {"annotator": a, "keyword": k, "label": "GAP", "novelty": 5.0, "keyword_set": "S", "rationale": ""}
            for k in ("k1", "k2")
            for a in ("a", "b")
        ]
        assert np.isnan(fleiss_kappa(pd.DataFrame(rows)))

    def test_krippendorff_alpha_is_high_for_consistent_scores(self, frame: pd.DataFrame) -> None:
        alpha = krippendorff_alpha_ordinal(frame)
        assert np.isnan(alpha) or alpha > 0.5

    def test_pairwise_agreement_covers_every_pair(self, frame: pd.DataFrame) -> None:
        pairs = pairwise_agreement(frame)
        assert len(pairs) == 3
        assert pairs["label_agreement"].between(0, 1).all()
        assert pairs.loc[pairs["annotator_a"] == "a", "n_keywords"].iloc[0] == 4

    def test_compare_keyword_sets_quantifies_the_filter_effect(self) -> None:
        selected = pd.DataFrame(
            {
                "annotator": ["a"] * 4,
                "keyword_set": ["SELECTED"] * 4,
                "keyword": list("abcd"),
                "label": ["GAP", "GAP", "GAP", "EXPLORED"],
                "novelty": [7.0, 8.0, 6.0, 2.0],
                "rationale": [""] * 4,
            }
        )
        everything = pd.DataFrame(
            {
                "annotator": ["a"] * 4,
                "keyword_set": ["ALL"] * 4,
                "keyword": list("efgh"),
                "label": ["GAP", "EXPLORED", "EXPLORED", "METHOD"],
                "novelty": [5.0, 2.0, 1.0, 1.0],
                "rationale": [""] * 4,
            }
        )
        comparison = compare_keyword_sets(selected, everything)
        assert comparison.selected_rates["GAP"] == pytest.approx(0.75)
        assert comparison.all_rates["GAP"] == pytest.approx(0.25)
        assert comparison.gap_rate_delta_pp == pytest.approx(50.0)
        assert comparison.explored_rate_delta_pp == pytest.approx(-25.0)
        assert comparison.novelty_increase_pct > 0
        assert set(comparison.novelty_test) == {
            "overlapping_samples", "welch_t", "welch_p", "mannwhitney_u",
            "mannwhitney_p", "cohens_d", "permutation_p",
        }
        assert comparison.novelty_test["overlapping_samples"] == 0.0
        assert "gap_rate_delta_pp" in comparison.to_dict()


class TestExpertConcordance:
    @pytest.fixture()
    def llm_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "annotator": ["m1"] * 5 + ["m2"] * 5,
                "keyword_set": ["SELECTED"] * 10,
                "keyword": list("abcde") * 2,
                "label": ["GAP", "GAP", "EXPLORED", "METHOD", "GAP"] * 2,
                "novelty": [8.0, 7.0, 2.0, 1.0, 6.0, 7.0, 6.0, 3.0, 2.0, 5.0],
                "rationale": [""] * 10,
            }
        )

    @pytest.fixture()
    def expert_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "expert": ["e1"] * 5 + ["e2"] * 5,
                "keyword": list("abcde") * 2,
                "novelty": [7.0, 6.0, 3.0, 2.0, 5.0, 9.0, 7.0, 2.0, 1.0, 6.0],
                "label": ["GAP", "GAP", "EXPLORED", "METHOD", "GAP"] * 2,
            }
        )

    def test_reports_spearman_and_label_agreement(self, llm_frame, expert_frame) -> None:
        report = expert_llm_concordance(llm_frame, expert_frame)
        assert report["n_shared_keywords"] == 5
        assert report["n_experts"] == 2
        assert report["spearman_rho"] > 0.8
        assert report["label_agreement"] == pytest.approx(1.0)

    def test_dispersion_of_experts_and_models_is_reported(self, llm_frame, expert_frame) -> None:
        report = expert_llm_concordance(llm_frame, expert_frame)
        assert report["expert_dispersion"] > 0
        assert report["llm_dispersion"] > 0

    def test_too_few_shared_keywords_raises(self, llm_frame) -> None:
        sparse = pd.DataFrame({"expert": ["e1"], "keyword": ["a"], "novelty": [5.0]})
        with pytest.raises(ValueError, match="at least 3 shared"):
            expert_llm_concordance(llm_frame, sparse)

    def test_missing_column_raises(self, llm_frame) -> None:
        with pytest.raises(ValueError, match="missing column"):
            expert_llm_concordance(llm_frame, pd.DataFrame({"keyword": ["a"], "novelty": [1.0]}))

    def test_expert_report_filters_by_keyword_set(self, llm_frame, expert_frame) -> None:
        assert expert_report(llm_frame, expert_frame)["n_shared_keywords"] == 5


class TestExpertSheets:
    def test_sheet_is_blinded_and_shuffled(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        sheet = build_expert_sheet(result, "problem statement")
        assert set(sheet.columns) == {"item", "keyword", "novelty", "label", "comment"}
        assert (sheet["novelty"] == "").all()
        assert len(sheet) == result.gaps["keyword"].nunique()

    def test_scores_can_be_included_deliberately(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        sheet = build_expert_sheet(result, "problem", include_scores=True, shuffle_seed=None)
        assert {"gap_type", "score"} <= set(sheet.columns)

    def test_round_trip_through_excel(self, synthetic_corpus, offline_config, tmp_path: Path) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        sheet = build_expert_sheet(result, "problem")
        path = write_expert_sheet(sheet, tmp_path / "experts.xlsx", "problem", ("e1", "e2"))
        book = pd.read_excel(path, sheet_name=None)
        assert set(book) == {"instructions", "e1", "e2"}

        completed = sheet.copy()
        completed["novelty"] = 6
        completed["label"] = "GAP"
        completed["expert"] = "e1"
        completed.to_csv(tmp_path / "e1.csv", index=False)
        loaded = load_expert_sheet(tmp_path / "e1.csv")
        assert set(loaded["expert"]) == {"e1"}
        assert len(loaded) == len(sheet)

    def test_unrated_rows_are_dropped_on_load(self, tmp_path: Path) -> None:
        pd.DataFrame(
            {"keyword": ["a", "b", "c"], "novelty": [5, None, 99], "label": ["GAP", "", "GAP"]}
        ).to_csv(tmp_path / "partial.csv", index=False)
        loaded = load_expert_sheet(tmp_path / "partial.csv", expert="e9")
        assert list(loaded["keyword"]) == ["a"]

    def test_empty_run_cannot_produce_a_sheet(self, synthetic_corpus, offline_config) -> None:
        config = dataclasses.replace(offline_config, gaps=GapConfig(min_tfidf=99.0, max_keyword_frequency=0))
        result = articles_first(synthetic_corpus, config)
        with pytest.raises(ValueError, match="no candidate gaps"):
            build_expert_sheet(result, "problem")


class TestValidationRunner:
    def test_selected_and_all_split(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        selected, everything = selected_and_all_keywords(result, max_all=10)
        assert selected and len(everything) == 10
        assert len(set(selected)) == len(selected)

    def test_control_set_cap_samples_rather_than_truncates(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        _, full = selected_and_all_keywords(result)
        _, capped = selected_and_all_keywords(result, max_all=8, random_state=1)
        _, capped_again = selected_and_all_keywords(result, max_all=8, random_state=1)
        _, other_seed = selected_and_all_keywords(result, max_all=8, random_state=2)
        assert len(capped) == 8
        assert set(capped) <= set(full)
        assert capped == capped_again              # deterministic for a fixed seed
        assert capped != other_seed                # and a sample, not the head
        assert capped != full[:8]

    def test_control_set_is_ordered_by_corpus_frequency(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        _, control = selected_and_all_keywords(result)
        counts = {
            keyword: sum(
                keyword in set(keywords)
                for keywords in result.articles["keywords_normalized"]
            )
            for keyword in control
        }
        frequencies = [counts[keyword] for keyword in control]
        assert frequencies == sorted(frequencies, reverse=True)
        assert set(control) == {
            keyword
            for keywords in result.articles["keywords_normalized"]
            for keyword in keywords
        }

    def test_control_annotations_can_be_reused(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        planted = set(result.gaps["keyword"])
        selected, control = selected_and_all_keywords(result, max_all=10)

        calls: list[str] = []

        def counting_stub(name: str):
            stub = make_stub(name, planted)
            inner = stub._completion

            def completion(prompt: str) -> str:
                calls.append(name)
                return inner(prompt)

            return CallableAnnotator(name, completion, batch_size=50)

        panel = [counting_stub("m1"), counting_stub("m2")]
        prepared = [
            Annotation(keyword, "EXPLORED", 2.0, "prepared", keyword_set="ALL")
            for keyword in control
        ]
        report = validate_gaps(
            result, panel, "problem", "domain",
            selected=selected, all_keywords=control, all_annotations=prepared,
        )
        assert len(calls) == 2  # one SELECTED batch per annotator, no ALL calls
        assert set(report.annotations["annotator"]) == {"m1", "m2", "prepared"}
        assert report.comparison is not None
        assert report.comparison.all_rates["EXPLORED"] == pytest.approx(1.0)

    def test_report_recomputes_without_annotators(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        selected, control = selected_and_all_keywords(result, max_all=10)
        report = validate_gaps(
            result, [], "problem", "domain",
            selected=selected, all_keywords=control,
            selected_annotations=[
                Annotation(kw, "GAP", 8.0, "stored", keyword_set="SELECTED")
                for kw in selected
            ],
            all_annotations=[
                Annotation(kw, "EXPLORED", 2.0, "stored", keyword_set="ALL")
                for kw in control
            ],
        )
        assert report.comparison is not None
        assert report.comparison.selected_rates["GAP"] == pytest.approx(1.0)
        assert report.comparison.all_rates["GAP"] == pytest.approx(0.0)
        assert report.failures == {}

    def test_empty_panel_without_stored_annotations_raises(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        with pytest.raises(ValueError, match="no annotators supplied"):
            validate_gaps(result, [], "problem", "domain")

    def test_panel_produces_a_full_report(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        planted = set(result.gaps["keyword"])
        panel = [make_stub(f"model{i}", planted) for i in range(3)]
        report = validate_gaps(result, panel, "How do these themes interact?", "test domain", 12)
        assert report.annotators == ["model0", "model1", "model2"]
        assert report.comparison is not None
        assert report.comparison.gap_rate_delta_pp > 0
        assert not report.consensus.empty
        assert report.failures == {}

    def test_failing_annotator_is_recorded_not_fatal(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        planted = set(result.gaps["keyword"])

        def broken(prompt: str) -> str:
            raise RuntimeError("transport error")

        report = validate_gaps(
            result,
            [make_stub("good", planted), CallableAnnotator("broken", broken)],
            "problem", "domain", 8,
        )
        assert report.annotators == ["good"]
        assert any("broken" in key for key in report.failures)

    def test_all_annotators_failing_raises(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)

        def broken(prompt: str) -> str:
            raise RuntimeError("down")

        with pytest.raises(RuntimeError, match="every annotator failed"):
            validate_gaps(result, [CallableAnnotator("broken", broken)], "problem")

    def test_no_annotators_raises(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        with pytest.raises(ValueError, match="no annotators"):
            validate_gaps(result, [], "problem")

    def test_report_is_saved(self, synthetic_corpus, offline_config, tmp_path: Path) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        report = validate_gaps(
            result, [make_stub("m", set(result.gaps["keyword"]))], "problem", "domain", 8
        )
        written = report.save(tmp_path)
        for key in ("annotations", "consensus", "agreement", "summary"):
            assert Path(written[key]).exists()
        payload = json.loads(Path(written["summary"]).read_text())
        assert payload["annotators"] == ["m"]


class TestMalformedReplies:
    KEYWORDS = ["alpha term", "beta term", "gamma term"]

    def parse(self, text: str):
        return parse_annotation_payload(text, self.KEYWORDS, "m1", "SELECTED")

    def test_unescaped_quote_loses_only_its_own_entry(self) -> None:
        text = (
            '{"annotations": ['
            '{"keyword": "alpha term", "novelty": 7, "label": "GAP", "rationale": "ok"},'
            '{"keyword": "beta term", "novelty": 2, "label": "EXPLORED", '
            '"rationale": "the "beta" line is broken"},'
            '{"keyword": "gamma term", "novelty": 5, "label": "GAP", "rationale": "fine"}'
            "]}"
        )
        annotations = self.parse(text)
        assert [a.keyword for a in annotations] == ["alpha term", "gamma term"]
        assert annotations[0].novelty == 7.0

    def test_trailing_comma_is_repaired(self) -> None:
        text = (
            '{"annotations": ['
            '{"keyword": "alpha term", "novelty": 4, "label": "GAP", "rationale": "ok",},'
            '{"keyword": "beta term", "novelty": 1, "label": "NOISE", "rationale": "no",}'
            "]}"
        )
        assert {a.keyword for a in self.parse(text)} == {"alpha term", "beta term"}

    def test_truncated_reply_keeps_complete_entries(self) -> None:
        text = (
            '{"annotations": ['
            '{"keyword": "alpha term", "novelty": 6, "label": "GAP", "rationale": "ok"},'
            '{"keyword": "beta term", "novelty": 3, "label": "EXPL'
        )
        annotations = self.parse(text)
        assert [a.keyword for a in annotations] == ["alpha term"]

    def test_unsalvageable_reply_raises(self) -> None:
        with pytest.raises(ValueError, match="no entry could be salvaged"):
            self.parse('{"annotations": [{"novelty": 3, "label": "GAP",}], "note": "x",}')

    def test_reply_without_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            self.parse("I am unable to comply with this request.")

    def test_batch_is_retried_once(self) -> None:
        replies = iter([
            "not json at all",
            '{"annotations": [{"keyword": "alpha term", "novelty": 8, '
            '"label": "GAP", "rationale": "ok"}]}',
        ])
        calls: list[str] = []

        def completion(prompt: str) -> str:
            calls.append(prompt)
            return next(replies)

        annotator = CallableAnnotator("retrying", completion, batch_size=10)
        annotations = annotator.annotate(["alpha term"], "problem", "domain", "SELECTED")
        assert len(calls) == 2
        assert [a.keyword for a in annotations] == ["alpha term"]

    def test_persistently_bad_replies_raise(self) -> None:
        annotator = CallableAnnotator("broken", lambda prompt: "nonsense", batch_size=10)
        with pytest.raises(ValueError, match="no JSON object"):
            annotator.annotate(["alpha term"], "problem", "domain", "SELECTED")


def test_annotations_to_frame_schema() -> None:
    frame = annotations_to_frame([Annotation("alpha term", "GAP", 7, "m1", "why", "SELECTED")])
    assert list(frame.columns) == [
        "annotator", "keyword_set", "keyword", "label", "novelty", "rationale"
    ]
    assert annotations_to_frame([]).empty


class TestControlSetIndependence:
    """The control set must not contain the candidates it is compared against."""

    def test_control_excludes_the_candidates(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        selected, control = selected_and_control_keywords(result)
        assert selected, "fixture must produce candidates"
        assert not set(selected) & set(control)

    def test_control_is_the_rest_of_the_keyword_population(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        selected, control = selected_and_control_keywords(result)
        population = {kw for kws in result.articles["keywords_normalized"] for kw in kws}
        assert set(control) == population - set(selected)

    def test_nested_control_is_available_but_deprecated(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        with pytest.deprecated_call():
            selected, nested = selected_and_all_keywords(result)
        assert set(selected) <= set(nested)

    def test_cap_samples_from_the_disjoint_control(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        selected, control = selected_and_control_keywords(
            result, max_control=8, random_state=1
        )
        assert len(control) == 8
        assert not set(selected) & set(control)

    def test_overlapping_samples_suppress_the_parametric_tests(self) -> None:
        shared = pd.DataFrame({
            "annotator": ["m1"] * 4, "keyword_set": ["SELECTED"] * 4,
            "keyword": ["a", "b", "c", "d"], "label": ["GAP"] * 4,
            "novelty": [5, 6, 7, 8], "rationale": [""] * 4,
        })
        nested = shared.assign(keyword_set="ALL")
        comparison = compare_keyword_sets(shared, nested)
        assert comparison.novelty_test["overlapping_samples"] == 1.0
        assert comparison.novelty_test["n_shared_keywords"] == 4.0
        assert "welch_p" not in comparison.novelty_test

    def test_disjoint_samples_get_all_three_tests(self) -> None:
        left = pd.DataFrame({
            "annotator": ["m1"] * 4, "keyword_set": ["SELECTED"] * 4,
            "keyword": ["a", "b", "c", "d"], "label": ["GAP"] * 4,
            "novelty": [8, 9, 8, 9], "rationale": [""] * 4,
        })
        right = pd.DataFrame({
            "annotator": ["m1"] * 4, "keyword_set": ["CONTROL"] * 4,
            "keyword": ["e", "f", "g", "h"], "label": ["EXPLORED"] * 4,
            "novelty": [1, 2, 1, 2], "rationale": [""] * 4,
        })
        comparison = compare_keyword_sets(left, right)
        assert comparison.novelty_test["overlapping_samples"] == 0.0
        for key in ("welch_p", "mannwhitney_p", "permutation_p", "cohens_d"):
            assert key in comparison.novelty_test
        assert comparison.novelty_test["permutation_p"] < 0.1


class TestPermutationTest:
    def test_identical_samples_are_never_significant(self) -> None:
        values = np.array([3.0, 4, 5, 3, 4, 5])
        assert permutation_mean_difference(values, values, n_resamples=2000) == 1.0

    def test_separated_samples_are_significant(self) -> None:
        p = permutation_mean_difference(
            np.arange(1.0, 11.0), np.arange(21.0, 31.0), n_resamples=2000
        )
        assert p < 0.01

    def test_p_value_is_never_zero(self) -> None:
        p = permutation_mean_difference(
            np.zeros(20), np.ones(20) * 9, n_resamples=500
        )
        assert 0 < p <= 1

    def test_deterministic_for_a_given_seed(self) -> None:
        left, right = np.array([2.0, 5, 7, 3]), np.array([4.0, 6, 1, 8])
        first = permutation_mean_difference(left, right, n_resamples=1000, random_state=7)
        second = permutation_mean_difference(left, right, n_resamples=1000, random_state=7)
        assert first == second

    def test_empty_input_is_undefined(self) -> None:
        assert np.isnan(permutation_mean_difference(np.array([]), np.array([1.0])))
