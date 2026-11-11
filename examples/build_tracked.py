"""Produce a change-marked copy of a revised document.

The marked copy is the revised document itself with the words that differ from
the submitted version given a yellow text background, and the words removed
from it shown struck through in grey.  Nothing is hidden: a reader of the
marked copy sees the revised text and can see what it replaced.

The marking is applied by splitting existing runs at the diff boundaries and
copying their run properties, rather than by rebuilding paragraphs, so bold,
italic and monospaced spans survive.

Usage::

    python build_tracked.py old.docx new.docx marked.docx
"""

from __future__ import annotations

import copy
import difflib
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import RGBColor

#: Words are the diff unit; whitespace is kept so offsets map back to the text.
TOKEN = re.compile(r"\S+\s*")

#: Word-sequence similarity below which a paragraph counts as rewritten rather
#: than edited, and is marked whole.
REWRITE_THRESHOLD = 0.60

#: Longest removed-text note worth printing after a revised sentence.
MAX_DELETION_CHARS = 120

#: Word-sequence similarity below which a new paragraph has no counterpart in
#: the submitted version and is reported as an addition rather than an edit.
PAIRING_FLOOR = 0.30


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)


#: Dash characters that were replaced wholesale for typographic reasons.  A
#: paragraph whose only difference is "—" against " - " did not change in any
#: sense a reviewer cares about, so the comparison folds them together; the
#: marked copy would otherwise be a wall of yellow that hides the real edits.
DASHES = str.maketrans({"\u2014": "-", "\u2013": "-", "\u2012": "-",
                        "\u2015": "-", "\u2212": "-", "\u2010": "-"})


def normalize(text: str) -> str:
    """Collapse whitespace and dash style, so neither counts as a change."""
    text = text.translate(DASHES)
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def inserted_ranges(old: str, new: str) -> list[tuple[int, int]]:
    """Character ranges of ``new`` that are not carried over from ``old``."""
    old_words, new_words = tokenize(old), tokenize(new)
    offsets, pos = [], 0
    for word in new_words:
        offsets.append((pos, pos + len(word)))
        pos += len(word)

    ranges: list[tuple[int, int]] = []
    matcher = difflib.SequenceMatcher(
        None, [w.strip() for w in old_words], [w.strip() for w in new_words],
        autojunk=False,
    )
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert") and j2 > j1:
            ranges.append((offsets[j1][0], offsets[j2 - 1][1]))
    return ranges


def deleted_text(old: str, new: str) -> list[tuple[int, str]]:
    """Words dropped from ``old``, with the position in ``new`` they sat at."""
    old_words, new_words = tokenize(old), tokenize(new)
    offsets, pos = [], 0
    for word in new_words:
        offsets.append(pos)
        pos += len(word)
    offsets.append(pos)

    out: list[tuple[int, str]] = []
    matcher = difflib.SequenceMatcher(
        None, [w.strip() for w in old_words], [w.strip() for w in new_words],
        autojunk=False,
    )
    for tag, i1, i2, j1, _ in matcher.get_opcodes():
        if tag in ("replace", "delete") and i2 > i1:
            out.append((offsets[min(j1, len(offsets) - 1)], "".join(old_words[i1:i2])))
    return out


def mark_paragraph(paragraph, old_text: str) -> int:
    """Highlight in ``paragraph`` whatever differs from ``old_text``."""
    new_text = paragraph.text
    if normalize(new_text) == normalize(old_text):
        return 0

    # A word-level diff of two paragraphs that share only their vocabulary
    # produces scattered single-word marks and an unreadable list of removed
    # fragments. Below the rewrite threshold the paragraph is simply marked
    # whole, which is both honest and legible; the submitted version remains
    # available for a side-by-side reading.
    similarity = difflib.SequenceMatcher(
        None, normalize(old_text).split(), normalize(new_text).split(), autojunk=False
    ).ratio()
    if similarity < REWRITE_THRESHOLD:
        highlight_whole(paragraph)
        return 1

    ranges = inserted_ranges(old_text, new_text)
    deletions = deleted_text(old_text, new_text)
    if not ranges and not deletions:
        return 0
    # Keep the removed-text note only when it is short enough to read.
    # Drop notes that carry no words -- heading numbers that moved into the
    # template's own list, stray punctuation -- which are noise, not changes.
    deletions = [(pos, t) for pos, t in deletions if re.search(r"[A-Za-z]{2}", t)]
    if sum(len(t) for _, t in deletions) > MAX_DELETION_CHARS:
        deletions = []

    pieces: list[tuple[str, object, bool]] = []  # text, rPr, highlight
    cursor = 0
    for run in list(paragraph.runs):
        text = run.text
        if not text:
            continue
        start, end = cursor, cursor + len(text)
        cuts = {start, end}
        for a, b in ranges:
            if a < end and b > start:
                cuts |= {max(a, start), min(b, end)}
        for point in sorted(cuts):
            pass
        bounds = sorted(cuts)
        for a, b in zip(bounds, bounds[1:]):
            if b <= a:
                continue
            marked = any(ra <= a and b <= rb for ra, rb in ranges)
            pieces.append((text[a - start:b - start], run._element.get_or_add_rPr(), marked))
        cursor = end

    for run in list(paragraph.runs):
        run._element.getparent().remove(run._element)

    for text, rpr, marked in pieces:
        run = paragraph.add_run()
        run._element.insert(0, copy.deepcopy(rpr))
        run.text = text
        if marked:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    # Deletions are appended at the end of the paragraph so the revised
    # sentence stays readable; each is struck through and greyed.
    if deletions:
        run = paragraph.add_run("  [removed: ")
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        run.font.size = paragraph.runs[0].font.size
        for _, text in deletions:
            run = paragraph.add_run(normalize(text) + " ")
            run.font.strike = True
            run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
            run.font.size = paragraph.runs[0].font.size
        run = paragraph.add_run("]")
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    return 1


def highlight_whole(paragraph) -> None:
    for run in paragraph.runs:
        if run.text.strip():
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def paragraph_texts(document) -> list[str]:
    return [p.text for p in document.paragraphs]


def table_texts(document) -> list[list[str]]:
    return [[c.text for r in t.rows for c in r.cells] for t in document.tables]


def build(old_path: str, new_path: str, out_path: str) -> dict[str, int]:
    old_doc, new_doc = Document(old_path), Document(new_path)
    marked = Document(new_path)

    old_pars = [normalize(t) for t in paragraph_texts(old_doc)]
    new_pars = [normalize(t) for t in paragraph_texts(new_doc)]
    matcher = difflib.SequenceMatcher(None, old_pars, new_pars, autojunk=False)

    changed = added = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            for j in range(j1, j2):
                if marked.paragraphs[j].text.strip():
                    highlight_whole(marked.paragraphs[j])
                    added += 1
        elif tag == "replace":
            # Pair by content, not by position. Inserting a paragraph shifts
            # every later index inside the block, so a positional pairing
            # compares unrelated paragraphs, scores them below the rewrite
            # threshold and marks both whole -- which turns an edited document
            # into a wall of yellow. Each new paragraph is therefore matched to
            # its most similar unused counterpart in the replaced old block,
            # and only counts as an addition when nothing there resembles it.
            candidates = list(range(i1, i2))
            taken: set[int] = set()
            for j in range(j1, j2):
                target = normalize(marked.paragraphs[j].text)
                if not target:
                    continue
                best, best_score = None, 0.0
                for i in candidates:
                    if i in taken:
                        continue
                    score = difflib.SequenceMatcher(
                        None, old_pars[i].split(), target.split(), autojunk=False
                    ).ratio()
                    if score > best_score:
                        best, best_score = i, score
                if best is not None and best_score >= PAIRING_FLOOR:
                    taken.add(best)
                    changed += mark_paragraph(
                        marked.paragraphs[j], old_doc.paragraphs[best].text
                    )
                else:
                    highlight_whole(marked.paragraphs[j])
                    added += 1

    # Tables: compare cell by cell where a counterpart exists, mark wholly new
    # tables in full.
    # Tables are matched by content, not by position: inserting or moving a
    # section shifts every later index, and a positional match would then
    # report whole unchanged tables as new.
    old_tables, new_tables = table_texts(old_doc), table_texts(new_doc)
    cells_changed = 0
    used: set[int] = set()
    for table, new_cells in zip(marked.tables, new_tables):
        best, best_score = None, 0.0
        for index, old_cells in enumerate(old_tables):
            if index in used:
                continue
            score = difflib.SequenceMatcher(
                None, [normalize(c) for c in old_cells],
                [normalize(c) for c in new_cells], autojunk=False,
            ).ratio()
            if score > best_score:
                best, best_score = index, score
        if best is None or best_score < 0.5:
            for row in table.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        highlight_whole(par)
            cells_changed += 1
            continue
        used.add(best)
        old_cells = old_tables[best]
        flat = [c for r in table.rows for c in r.cells]
        for position, cell in enumerate(flat):
            old_text = old_cells[position] if position < len(old_cells) else ""
            if normalize(cell.text) != normalize(old_text):
                for par in cell.paragraphs:
                    highlight_whole(par)
                cells_changed += 1

    marked.save(out_path)
    return {"paragraphs_changed": changed, "paragraphs_added": added,
            "table_cells_marked": cells_changed,
            "paragraphs_total": len(new_pars)}


if __name__ == "__main__":
    stats = build(*sys.argv[1:4])
    print(f"{sys.argv[3]}: {stats}", flush=True)
