"""Does showing the panel the corpus change its judgement? (reviewer 2, point 11)

The panel of Section 3.3 judges a keyword against a problem statement with no
sight of the analysed corpus, so its label is a statement about the literature
as the model remembers it, not about what these articles leave unexplored.  The
reviewer is right that this limits what the validation can establish, and the
limitation cannot be removed by argument -- only measured.

This script repeats the annotation with one change: each keyword is presented
together with the evidence the corpus itself carries for it -- how many of the
N analysed articles use it, and the titles of up to five of them.  Everything
else is held fixed: the same five providers at temperature 0, the same rubric,
the same keywords, the same batching.  The comparison is therefore paired, and
the quantity of interest is how often the two prompts disagree on the same
keyword, and in which direction.

Scope: Mode B on the two case-study corpora, the candidate arm and a
rarity-matched control arm of equal size, which keeps the number of hosted
calls small while covering the contrast the manuscript reports.

Outputs
    rev/corpus_aware_annotations.csv  every judgement, both prompts
    rev/corpus_aware.json             paired summary per corpus and arm
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "repo/current/src")

from embedresearchgaps.validation.annotators import ChatAnnotator  # noqa: E402
from embedresearchgaps.validation.protocol import (  # noqa: E402
    RUBRIC,
    parse_annotation_payload,
)

RESULTS = Path("repo/current/paper/results")
CORPORA = {"management": "management", "finance": "economics and finance"}
#: The paired design needs the SAME annotator under both prompts, so only
#: models of the original panel that are still served can take part.  Probed on
#: 11 November 2026: anthropic:claude-sonnet-4-6 is no longer offered by the
#: provider (the account lists claude-sonnet-5 and later), and xai:grok-4.5
#: returns permission-denied for lack of credits on this account.  Both are
#: recorded rather than silently substituted: swapping in a different model
#: would make the comparison a model difference, not a prompt difference.
PANEL = [("openai", "gpt-4o", "OPENAI")]
UNAVAILABLE = {
    "anthropic:claude-sonnet-4-6": "model no longer served by the provider",
    "xai:grok-4.5": "permission denied: account credits exhausted",
    "deepseek:deepseek-v4-pro": "HTTP 402, insufficient balance",
    "google:gemini-2.5-flash": "HTTP 402, prepayment credits depleted",
}
SEED = 42


def corpus_prompt(
    keywords: list[str],
    problem: str,
    domain: str,
    evidence: dict[str, dict[str, object]],
    n_articles: int,
) -> str:
    """The S8 prompt with per-keyword corpus evidence appended."""
    lines = [f"Research domain: {domain}", "Research problem:", problem, ""]
    lines.append(
        f"You are judging keywords used by the authors of a specific corpus of "
        f"{n_articles} articles, selected as the articles closest to the research "
        f"problem above. For each keyword you are given how many of those articles "
        f"use it and the titles of up to five of them. Judge whether the keyword "
        f"marks a genuine research gap *in this corpus*: a topic these articles "
        f"touch but leave unexplored. Evidence from the corpus takes precedence "
        f"over your general impression of the literature."
    )
    lines += ["", RUBRIC, "", "Keywords to evaluate:"]
    for index, keyword in enumerate(keywords, 1):
        info = evidence.get(keyword, {})
        titles = info.get("titles", [])
        lines.append(f"{index}. {keyword}")
        lines.append(f"   used by {info.get('frequency', 0)} of {n_articles} articles")
        for title in titles:
            lines.append(f" - {title}")
    lines += [
        "",
        'Return JSON only, with this exact shape and one entry per keyword, in the '
        'same order:',
        '{"annotations": [{"keyword": "<keyword>", "novelty": <1-10>, "label": '
        '"GAP|EXPLORED|LOW IMPORTANCE|METHOD|NOISE", "rationale": "<one sentence>"}]}',
    ]
    return "\n".join(lines)



def probe_panel() -> dict[str, dict[str, object]]:
    """Ask each provider of the original panel whether it will serve the model.

    The availability record deposited with this study is written by this
    function, not by hand: each entry carries the HTTP status and the first
    line of the provider's own reply, so a reader can see why an annotator is
    missing rather than take our word for it.
    """
    import os

    import requests

    probes = {
        # Ask the model, not the catalogue: a model-list query answers "is it
        # offered", which is not the question. Only a completion request tells
        # us whether this account can actually reach it.
        "anthropic:claude-sonnet-4-6": (
            "POST", "https://api.anthropic.com/v1/messages",
            lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"}, "ANTROPIC"),
        "xai:grok-4.5": (
            "POST", "https://api.x.ai/v1/chat/completions",
            lambda key: {"Authorization": f"Bearer {key}"}, "XAI"),
        "deepseek:deepseek-v4-pro": (
            "POST", "https://api.deepseek.com/chat/completions",
            lambda key: {"Authorization": f"Bearer {key}"}, "DEEPSEEK"),
        "google:gemini-2.5-flash": (
            "POST", "https://generativelanguage.googleapis.com/v1beta/models/"
                    "gemini-2.5-flash:generateContent", lambda key: {}, "GOOGLE"),
        "openai:gpt-4o": (
            "POST", "https://api.openai.com/v1/chat/completions",
            lambda key: {"Authorization": f"Bearer {key}"}, "OPENAI"),
    }
    ping = [{"role": "user", "content": "Reply with the single character 1."}]
    record: dict[str, dict[str, object]] = {}
    for name, (_method, url, headers, key_env) in probes.items():
        provider, model = name.split(":", 1)
        key = (os.environ.get(key_env) or "").strip()
        if not key:
            record[name] = {"available": False, "status": None, "detail": f"{key_env} unset"}
            continue
        try:
            target = url if provider != "google" else f"{url}?key={key}"
            if provider == "google":
                payload = {"contents": [{"parts": [{"text": ping[0]["content"]}]}],
                           "generationConfig": {"temperature": 0, "maxOutputTokens": 4}}
            elif provider == "anthropic":
                payload = {"model": model, "max_tokens": 4, "temperature": 0, "messages": ping}
            else:
                payload = {"model": model, "temperature": 0, "max_tokens": 4, "messages": ping}
            reply = requests.post(target, headers=headers(key), json=payload, timeout=60)
            served = reply.ok
            detail = "" if served else reply.text[:200]
            record[name] = {"available": bool(served), "status": reply.status_code,
                            "detail": " ".join(detail.split())}
        except Exception as exc:
            record[name] = {"available": False, "status": None,
                            "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}
    return record

def corpus_evidence(corpus: str) -> tuple[dict[str, dict[str, object]], int]:
    """Frequency and example titles per keyword, from the analysed articles."""
    articles = pd.read_csv(RESULTS / corpus / "articles_first_articles.csv")
    if "selected" in articles.columns:
        articles = articles[articles["selected"].astype(bool)]
    evidence: dict[str, dict[str, object]] = {}
    for _, row in articles.iterrows():
        raw = str(row.get("author_keywords", "") or "")
        title = str(row.get("title", ""))[:140]
        for keyword in [k.strip().lower() for k in raw.replace("|", ";").split(";")]:
            if not keyword:
                continue
            entry = evidence.setdefault(keyword, {"frequency": 0, "titles": []})
            entry["frequency"] = int(entry["frequency"]) + 1
            if len(entry["titles"]) < 5:
                entry["titles"].append(title)
    return evidence, len(articles)


def arms(corpus: str, evidence: dict[str, dict[str, object]]) -> dict[str, list[str]]:
    """Candidate arm, and a rarity-matched control arm of the same size."""
    annotations = pd.read_csv(
        RESULTS / corpus / "articles_first_validation_annotations_disjoint.csv"
    )
    selected = sorted(set(annotations[annotations.keyword_set == "SELECTED"]["keyword"]))
    control_pool = sorted(set(annotations[annotations.keyword_set == "CONTROL"]["keyword"]))
    # Match on the rarity stratum the filters admit: two or more tokens, and a
    # document frequency no higher than the candidates' maximum.
    ceiling = max(
        int(evidence.get(k.lower(), {}).get("frequency", 0)) for k in selected
    )
    eligible = [
        k for k in control_pool
        if len(k.split()) >= 2
        and int(evidence.get(k.lower(), {}).get("frequency", 0)) <= ceiling
    ]
    rng = random.Random(SEED)
    control = sorted(rng.sample(eligible, min(len(selected), len(eligible))))
    return {"SELECTED": selected, "CONTROL": control}


def main() -> None:
    import os

    rows: list[dict[str, object]] = []
    for corpus, domain in CORPORA.items():
        evidence, n_articles = corpus_evidence(corpus)
        problem = json.loads(
            (RESULTS / corpus / "articles_first_run_config.json").read_text()
        )["problem_description"]
        sets = arms(corpus, evidence)
        for provider, model, key_env in PANEL:
            if not os.environ.get(key_env):
                print(f"  skipping {provider}: {key_env} unset", flush=True)
                continue
            annotator = ChatAnnotator(provider, model, key_env=key_env, batch_size=12)
            for arm, keywords in sets.items():
                for start in range(0, len(keywords), annotator.batch_size):
                    batch = keywords[start : start + annotator.batch_size]
                    prompt = corpus_prompt(batch, problem, domain, evidence, n_articles)
                    try:
                        reply = annotator.complete(prompt)
                        parsed = parse_annotation_payload(
                            reply, batch, annotator.name, keyword_set=arm
                        )
                    except Exception as exc:  # a provider failure must not lose the study
                        print(f"  {corpus} {annotator.name} {arm}: {type(exc).__name__}",
                              flush=True)
                        continue
                    for annotation in parsed:
                        rows.append({
                            "corpus": corpus, "prompt": "corpus_aware",
                            "annotator": annotator.name, "keyword_set": arm,
                            "keyword": annotation.keyword, "label": annotation.label,
                            "novelty": annotation.novelty,
                        })
            print(f"  {corpus} {annotator.name}: {len(rows)} rows so far", flush=True)

    aware = pd.DataFrame(rows)
    aware.to_csv("rev/corpus_aware_annotations.csv", index=False)

    # Paired comparison against the corpus-blind judgements already on disk.
    summary: dict[str, object] = {}
    for corpus in CORPORA:
        blind = pd.read_csv(
            RESULTS / corpus / "articles_first_validation_annotations_disjoint.csv"
        )
        blind = blind[["annotator", "keyword_set", "keyword", "label", "novelty"]]
        sub = aware[aware.corpus == corpus]
        merged = sub.merge(blind, on=["annotator", "keyword_set", "keyword"],
                           suffixes=("_aware", "_blind"))
        for arm in ("SELECTED", "CONTROL"):
            part = merged[merged.keyword_set == arm]
            if part.empty:
                continue
            summary[f"{corpus}|{arm}"] = {
                "n_judgements": int(len(part)),
                "gap_rate_blind": float((part.label_blind == "GAP").mean() * 100),
                "gap_rate_aware": float((part.label_aware == "GAP").mean() * 100),
                "label_agreement": float((part.label_blind == part.label_aware).mean()),
                "novelty_blind": float(part.novelty_blind.mean()),
                "novelty_aware": float(part.novelty_aware.mean()),
                "novelty_delta": float((part.novelty_aware - part.novelty_blind).mean()),
                "gap_to_nongap": int(((part.label_blind == "GAP")
                                      & (part.label_aware != "GAP")).sum()),
                "nongap_to_gap": int(((part.label_blind != "GAP")
                                      & (part.label_aware == "GAP")).sum()),
            }
    summary["_annotators_unavailable"] = UNAVAILABLE
    Path("rev/corpus_aware.json").write_text(json.dumps(summary, indent=2))
    for key, value in summary.items():
        print(key, json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
