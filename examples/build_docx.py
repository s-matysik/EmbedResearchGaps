"""Fill the SoftwareX Original Software Publication template with the manuscript.

The template is opened as the base document, so every format setting it carries
-- page size and margins, the Calibri 11 pt document default, the Cambria
heading styles, the paragraph spacing, and the code-metadata table with its
borders -- is inherited rather than recreated.  The template's instruction
paragraphs are removed and its mandatory section skeleton is filled from
``manuscript/manuscript.md``; the template's own table element is reused for the
code metadata, with only its third column written.

Markdown handled: headings, paragraphs, pipe tables, ``![caption](path)``
figures, ``^a^`` superscripts and inline bold / italic / code.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml.shared import OxmlElement
from docx.shared import Inches, Pt

TEMPLATE = Path("/Users/sebastianmatysik/!!!!EmbedResearchGaps/softwarex-osp-template-.docx")
SOURCE = Path("manuscript/manuscript.md")
TARGET = Path("manuscript/EmbedResearchGaps_SoftwareX.docx")

#: Usable text width: Letter minus the template's one-inch margins.
TEXT_WIDTH_IN = 6.5

#: The five mandatory section headings, in the template's own wording.  The
#: template carries them as Heading 1 paragraphs attached to its multilevel
#: list (numId 1: "%1." at level 0, "%1.%2." at level 1), so Word generates
#: "1.", "2.", "2.1." itself.  Heading text must therefore be unnumbered: a
#: number typed into the text would be duplicated by the list.
SECTION_TITLES = {
    "1. Motivation and significance": "Motivation and significance",
    "2. Software description": "Software description",
    "2.1 Software architecture": "Software architecture",
    "2.2 Software functionalities": "Software functionalities",
    "3. Illustrative examples": "Illustrative examples",
    "3.1 What the modes return": "What the modes return",
    "3.2 How stable the candidate lists are": "How stable the candidate lists are",
    "3.3 What the panel says": "What the panel says",
    "4. Impact": "Impact",
    "5. Conclusions": "Conclusions",
}

#: The list the template attaches to its section headings.
SECTION_LIST_ID = 1

INLINE = re.compile(r"(\*\*.*?\*\*|\*[^*]+?\*|`[^`]+?`|\^[^^]+?\^|\[.*?\]\(.*?\))", re.S)


def blank_document() -> Document:
    """Open the template and strip its body, keeping styles and page setup."""
    document = Document(str(TEMPLATE))
    metadata_table = document.tables[0]._tbl
    body = document.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)
    return document, metadata_table


def add_inline(paragraph, text: str) -> None:
    """Add ``text`` to ``paragraph``, honouring inline markdown."""
    text = text.replace("\\*", "*").replace("\\_", "_")
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            paragraph.add_run(piece[2:-2]).bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = paragraph.add_run(piece[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(10)
        elif piece.startswith("^") and piece.endswith("^"):
            paragraph.add_run(piece[1:-1].replace("\\*", "*")).font.superscript = True
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True
        elif piece.startswith("["):
            match = re.fullmatch(r"\[(.*?)\]\((.*?)\)", piece, re.S)
            paragraph.add_run(match.group(1) or match.group(2) if match else piece)
        else:
            paragraph.add_run(piece)


def heading(document: Document, text: str, level: int = 1, numbered: bool = False):
    """Add a heading, optionally attached to the template's section list.

    ``numbered`` attaches the paragraph to numId 1 at ``level - 1``, which is
    how the template itself numbers its five mandatory sections and their
    subsections.  Front matter, the declarations and the reference list are
    unnumbered in the template and are added without it.
    """
    paragraph = document.add_paragraph(text, style=f"Heading {level}")
    if numbered:
        properties = paragraph._p.get_or_add_pPr()
        number = OxmlElement("w:numPr")
        for tag, value in (("w:ilvl", level - 1), ("w:numId", SECTION_LIST_ID)):
            element = OxmlElement(tag)
            element.set(qn("w:val"), str(value))
            number.append(element)
        properties.append(number)
    return paragraph


def prose(document: Document, text: str):
    paragraph = document.add_paragraph(style="Body")
    add_inline(paragraph, " ".join(text.split()))
    return paragraph


def fill_metadata_table(table_element, rows: list[list[str]]) -> None:
    """Write the manuscript's metadata into the template's own table."""
    from docx.table import Table

    table = Table(table_element, None)
    assert len(table.rows) - 1 == len(rows), (len(table.rows), len(rows))
    for source, row in zip(rows, table.rows[1:]):
        assert source[0] == row.cells[0].text.strip(), (source[0], row.cells[0].text)
        cell = row.cells[2]
        cell.text = ""
        add_inline(cell.paragraphs[0], source[2])
        for run in cell.paragraphs[0].runs:
            run.italic = False


def parse_blocks(text: str):
    """Split the manuscript body into (kind, payload) blocks."""
    blocks: list[tuple[str, object]] = []
    buffer: list[str] = []
    table: list[list[str]] = []

    def flush() -> None:
        if buffer:
            blocks.append(("p", " ".join(buffer)))
            buffer.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("table", [r for r in table]))
            table.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            flush()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                table.append(cells)
            continue
        flush_table()
        if not stripped:
            flush()
        elif stripped.startswith("!["):
            flush()
            match = re.fullmatch(r"!\[(.*)\]\((.*)\)", stripped, re.S)
            blocks.append(("figure", (match.group(1), match.group(2))))
        elif stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append((f"h{level}", stripped.lstrip("# ")))
        else:
            buffer.append(stripped)
    flush()
    flush_table()
    return blocks


def build() -> Path:
    text = SOURCE.read_text(encoding="utf-8")
    document, metadata_table = blank_document()
    blocks = parse_blocks(text)

    title = next(p for k, p in blocks if k == "h1")

    # --- Title, authors, affiliations -----------------------------------
    heading(document, title)
    index = [k for k, _ in blocks].index("h1")
    author_line, affiliations, corresponding = None, [], None
    for kind, payload in blocks[index + 1:]:
        if kind != "p":
            break
        if author_line is None:
            author_line = payload
        elif payload.startswith("\\*"):
            corresponding = payload
        else:
            affiliations.append(payload)
    paragraph = document.add_paragraph(style="Body")
    add_inline(paragraph, author_line)
    # The affiliation lines carry no blank line between them, so they arrive as
    # one buffered block; re-split on the superscript markers that open each.
    for piece in re.findall(r"\^[a-z]\^[^^]*", " ".join(affiliations)):
        prose(document, piece.strip())
    if corresponding:
        prose(document, corresponding)

    # --- Abstract, keywords ---------------------------------------------
    body_text = text.split("## Abstract", 1)[1]
    abstract = body_text.split("**Keywords:**", 1)[0].strip()
    keywords = body_text.split("**Keywords:**", 1)[1].split("## Metadata", 1)[0].strip()
    heading(document, "Abstract")
    prose(document, abstract)
    heading(document, "Keywords")
    prose(document, keywords)

    # --- Metadata table --------------------------------------------------
    heading(document, "Metadata")
    caption = document.add_paragraph(style="Body")
    add_inline(caption, "**Table 1.** Code metadata.")
    rows = next(p for k, p in blocks if k == "table")
    fill_metadata_table(metadata_table, rows[1:])
    document.paragraphs[-1]._p.addnext(metadata_table)
    document.add_paragraph(style="Body")

    # --- Body sections ---------------------------------------------------
    start = [i for i, (k, p) in enumerate(blocks)
             if k == "h2" and p.startswith("1. Motivation")][0]
    for kind, payload in blocks[start:]:
        if kind == "h2":
            heading(document, SECTION_TITLES.get(payload, payload), 1,
                    numbered=payload in SECTION_TITLES)
        elif kind == "h3":
            heading(document, SECTION_TITLES.get(payload, payload), 2,
                    numbered=payload in SECTION_TITLES)
        elif kind == "p":
            prose(document, payload)
        elif kind == "figure":
            caption_text, path = payload
            image = Path(path)
            if not image.exists():
                image = Path("manuscript") / image.name
            picture = document.add_paragraph(style="Body")
            picture.add_run().add_picture(str(image), width=Inches(TEXT_WIDTH_IN))
            figure_caption = document.add_paragraph(style="Body")
            add_inline(figure_caption, caption_text)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    document.save(TARGET)
    return TARGET


if __name__ == "__main__":
    path = build()
    built = Document(str(path))
    print(f"{path}: {len(built.paragraphs)} paragraphs, {len(built.tables)} table(s), "
          f"{len(built.inline_shapes)} images", flush=True)
