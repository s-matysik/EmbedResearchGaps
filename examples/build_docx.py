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

#: Section numbers are part of the heading text, as in the reference paper:
#: no list numbering is attached, so a number dropped from the text would
#: simply disappear. The markdown already carries them, so the map only fixes
#: the punctuation the journal uses for subsections ("2.1." rather than "2.1").
SECTION_TITLES = {
    "2.1 Software architecture": "2.1. Software architecture",
    "2.2 Software functionalities": "2.2. Software functionalities",
    "3.1 What the modes return": "3.1. What the modes return",
    "3.2 How stable the candidate lists are": "3.2. How stable the candidate lists are",
    "3.3 What the panel says": "3.3. What the panel says",
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
            run.font.size = Pt(SMALL_PT)
        elif piece.startswith("^") and piece.endswith("^"):
            paragraph.add_run(piece[1:-1].replace("\\*", "*")).font.superscript = True
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True
        elif piece.startswith("["):
            match = re.fullmatch(r"\[(.*?)\]\((.*?)\)", piece, re.S)
            paragraph.add_run(match.group(1) or match.group(2) if match else piece)
        else:
            paragraph.add_run(piece)


#: House formatting, taken from an accepted Original Software Publication of
#: the same group rather than from the template's style definitions: the whole
#: document is Arial, headings are 11 pt and not bold with no space above
#: them, captions and reference entries drop to 9 pt, and subsection headings
#: are indented by a quarter inch.  The template's heading styles (Cambria 14
#: and 13 pt) and its multilevel section list are deliberately not used: they
#: render section titles larger than the body and at a different typeface,
#: which is what made the earlier build look unlike the journal's own papers.
BODY_FONT = "Arial"
HEADING_PT = 11
SMALL_PT = 9
SUBSECTION_INDENT_IN = 0.25


def heading(document: Document, text: str, level: int = 1, numbered: bool = False):
    """Add a section heading in the house format.

    ``numbered`` is accepted for call compatibility and ignored: section
    numbers are part of the heading text, as in the reference paper, so no
    list numbering is attached.  Word would otherwise prepend its own number
    to a title that already carries one.
    """
    # The journal writes subsection numbers with a trailing dot ("2.1.
    # Software architecture"); the markdown source writes "2.1". Normalising
    # here keeps the two numbering shapes from appearing in one document.
    text = re.sub(r"^(\d+\.\d+)(?=\s)", r"\1.", text.strip())
    paragraph = document.add_paragraph(text, style="Heading 1")
    for run in paragraph.runs:
        run.font.name = BODY_FONT
        run.font.size = Pt(HEADING_PT)
        # Bold is left to the Heading 1 style, which supplies it: the
        # reference paper's heading runs carry no bold property of their own.
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    if level > 1:
        paragraph.paragraph_format.left_indent = Inches(SUBSECTION_INDENT_IN)
    return paragraph



def complete_run_fonts(document: Document) -> int:
    """Mirror each run's font onto the complex-script slots.

    python-docx writes only ``w:ascii`` and ``w:hAnsi``; Word also reads
    ``w:cs`` for complex-script runs and ``w:szCs`` for their size, and the
    reference paper sets all of them.  Where they are missing a renderer falls
    back to the theme font, which is how a document that looks like Arial on
    screen prints a different face in places.
    """
    fixed = 0
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            fixed += _complete_run(run)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        fixed += _complete_run(run)
    return fixed


def _complete_run(run) -> int:
    if not run.text.strip():
        return 0
    properties = run._element.get_or_add_rPr()
    fonts = properties.find(qn("w:rFonts"))
    changed = 0
    if fonts is not None:
        ascii_font = fonts.get(qn("w:ascii"))
        if ascii_font and fonts.get(qn("w:cs")) != ascii_font:
            fonts.set(qn("w:cs"), ascii_font)
            fonts.set(qn("w:eastAsia"), ascii_font)
            changed = 1
    size = properties.find(qn("w:sz"))
    if size is not None and properties.find(qn("w:szCs")) is None:
        size_cs = OxmlElement("w:szCs")
        size_cs.set(qn("w:val"), size.get(qn("w:val")))
        properties.append(size_cs)
        changed = 1
    return changed


CAPTION = re.compile(r"^(Figure|Table)\s+\d+\.")
CONTENT_INDENT_IN = 0.55


def _kind(paragraph) -> str:
    """Classify a paragraph the way the reference paper's formatting does."""
    text = paragraph.text.strip()
    if paragraph._p.xpath(".//w:drawing"):
        return "picture"
    if paragraph.style.name.startswith("Heading"):
        return "heading"
    runs = [r for r in paragraph.runs if r.text.strip()]
    if runs and all(r.font.name == "Consolas" for r in runs):
        return "code"
    if CAPTION.match(text):
        return "caption"
    if re.match(r"^\[\d+\]\s", text):
        return "reference"
    return "text" if text else "blank"


def apply_house_format(document: Document, *, indent_subsections: bool = True,
                       tables: bool = True) -> dict:
    """Apply the formatting of the reference paper to a built document.

    Every rule was read from the reference's own paragraphs, not from the
    journal template's style definitions:

    * headings: Arial 11 pt, bold from the style, 0 pt above, single spacing;
      subsection headings indented 0.25 in;
    * content under a subsection (text, captions, code) indented 0.55 in, and
      content under a top-level heading not indented at all;
    * running text and captions keep Word's default 10 pt after each paragraph,
      code blocks and reference entries sit tight (0 pt);
    * everything single spaced; captions and references 9 pt;
    * the metadata table's text is 9 pt, the results table is centred with
      centred cells.
    """
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    # Only a paper has front matter (title, authors) before its Abstract; a
    # supplement starts straight into running text.
    has_abstract = any(p.text.strip() == "Abstract" for p in document.paragraphs)
    in_subsection = in_references = False
    past_front_matter = not has_abstract
    counts: dict[str, int] = {}
    table_index = 0
    for child in list(document.element.body):
        tag = child.tag.split("}")[-1]
        if tag == "tbl":
            table = Table(child, document)
            if tables:
                for row in table.rows:
                    for cell in row.cells:
                        for par in cell.paragraphs:
                            for run in par.runs:
                                run.font.size = Pt(SMALL_PT)
                                run.font.name = BODY_FONT
                if table_index == 1:
                    table.alignment = WD_TABLE_ALIGNMENT.CENTER
                    for row in table.rows:
                        for cell in row.cells:
                            for par in cell.paragraphs:
                                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            table_index += 1
            continue
        if tag != "p":
            continue
        par = Paragraph(child, document)
        kind = _kind(par)
        counts[kind] = counts.get(kind, 0) + 1
        fmt = par.paragraph_format
        fmt.line_spacing = 1.0
        if kind == "heading":
            text = par.text.strip()
            past_front_matter = past_front_matter or text == "Abstract"
            in_references = text == "References"
            in_subsection = bool(indent_subsections and fmt.left_indent)
            fmt.space_before = Pt(0)
            continue
        if kind == "blank":
            continue
        if not past_front_matter:
            fmt.space_after = Pt(0)          # author and affiliation block
            continue
        if kind == "reference" or in_references:
            fmt.space_after = Pt(0)
            fmt.left_indent = None
        elif kind == "code":
            fmt.space_after = Pt(0)
            fmt.left_indent = Inches(CONTENT_INDENT_IN) if in_subsection else None
        elif kind == "picture":
            fmt.space_after = None
            fmt.left_indent = None
        else:  # running text and captions
            fmt.space_after = None           # inherit the default 10 pt
            fmt.left_indent = Inches(CONTENT_INDENT_IN) if in_subsection else None
    return counts


def caption_markup(text: str) -> str:
    """Rewrite "**Fig. 2.** ..." as "Figure 2. ...": the reference paper's
    captions spell the word out and do not set the label in bold."""
    text = re.sub(r"^\*\*Fig\. (\d+)\.\*\*", r"Figure \1.", text.strip())
    return re.sub(r"^\*\*Table (\d+)\.\*\*", r"Table \1.", text)


def prose(document: Document, text: str, size_pt: float | None = None):
    text = caption_markup(text)
    if CAPTION.match(text) and size_pt is None:
        size_pt = SMALL_PT
    paragraph = document.add_paragraph(style="Body")
    add_inline(paragraph, " ".join(text.split()))
    for run in paragraph.runs:
        if run.font.name is None:
            run.font.name = BODY_FONT
        if size_pt is not None:
            run.font.size = Pt(size_pt)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
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

    in_code = False
    code: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        # Fenced code: keep the raw line, indentation included, and emit it as
        # its own block. Joining these into a paragraph the way prose is joined
        # would destroy the line breaks and leave the fences in the document.
        if stripped.startswith("```"):
            if in_code:
                blocks.append(("code", list(code)))
                code.clear()
            else:
                flush()
                flush_table()
            in_code = not in_code
            continue
        if in_code:
            code.append(line.rstrip())
            continue
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
    # add_inline leaves the font to the style; the author block must carry the
    # document font explicitly like every other run, or it prints in Calibri.
    for run in paragraph.runs:
        if run.font.name is None:
            run.font.name = BODY_FONT
    paragraph.paragraph_format.line_spacing = 1.0
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
    add_inline(caption, "Table 1. Code metadata.")
    for run in caption.runs:
        if run.font.name is None:
            run.font.name = BODY_FONT
        run.font.size = Pt(SMALL_PT)
    rows = next(p for k, p in blocks if k == "table")
    blocks = [(k, p) for k, p in blocks if not (k == "table" and p is rows)]
    fill_metadata_table(metadata_table, rows[1:])
    document.paragraphs[-1]._p.addnext(metadata_table)
    document.add_paragraph(style="Body")

    # --- Body sections ---------------------------------------------------
    start = [i for i, (k, p) in enumerate(blocks)
             if k == "h2" and p.startswith("1. Motivation")][0]
    for kind, payload in blocks[start:]:
        if kind == "h2":
            heading(document, SECTION_TITLES.get(payload, payload), 1)
        elif kind == "h3":
            heading(document, SECTION_TITLES.get(payload, payload), 2)
        elif kind == "table":
            # A second markdown table in the body is a results table: build it
            # with the template's own table style so borders and fonts match
            # the code-metadata table the template ships.
            header, *body_rows = payload
            built = document.add_table(rows=len(payload), cols=len(header))
            try:
                built.style = document.tables[0].style
            except Exception:
                pass
            # The template sets its own table text in Arial; match it, or the
            # two tables in one document render in two different typefaces.
            table_font = "Arial"
            try:
                first = document.tables[0].rows[0].cells[0].paragraphs[0]
                table_font = next(
                    (r.font.name for r in first.runs if r.font.name), "Arial"
                )
            except Exception:
                pass
            for column, text in enumerate(header):
                run = built.rows[0].cells[column].paragraphs[0].add_run(text)
                run.bold = True
                run.font.size = Pt(9)
                run.font.name = table_font
            for row_index, row in enumerate(body_rows, start=1):
                for column, text in enumerate(row):
                    cell = built.rows[row_index].cells[column]
                    run = cell.paragraphs[0].add_run(text)
                    run.font.size = Pt(9)
                    run.font.name = table_font
            document.add_paragraph(style="Body")
        elif kind == "code":
            # One paragraph with line breaks, as the reference sets its code;
            # leading spaces become non-breaking so Word keeps the indentation.
            par = document.add_paragraph(style="Body")
            rows_ = []
            for source_line in payload:
                indent = len(source_line) - len(source_line.lstrip(" "))
                rows_.append("\u00a0" * indent + source_line.strip())
            run = par.add_run("\n".join(rows_))
            run.font.name = "Consolas"
            run.font.size = Pt(SMALL_PT)
        elif kind == "p":
            # Reference entries are set a step smaller, like the captions.
            is_reference = bool(re.match(r"\[\d+\]\s", payload.strip()))
            prose(document, payload, size_pt=SMALL_PT if is_reference else None)
        elif kind == "figure":
            caption_text, path = payload
            image = Path(path)
            if not image.exists():
                image = Path("manuscript") / image.name
            picture = document.add_paragraph(style="Body")
            picture.add_run().add_picture(str(image), width=Inches(TEXT_WIDTH_IN))
            figure_caption = document.add_paragraph(style="Body")
            add_inline(figure_caption, caption_markup(caption_text))
            # Captions sit a step below body text, as in the reference paper.
            for run in figure_caption.runs:
                if run.font.name is None:
                    run.font.name = BODY_FONT
                run.font.size = Pt(SMALL_PT)

    apply_house_format(document)
    complete_run_fonts(document)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    document.save(TARGET)
    return TARGET


if __name__ == "__main__":
    path = build()
    built = Document(str(path))
    print(f"{path}: {len(built.paragraphs)} paragraphs, {len(built.tables)} table(s), "
          f"{len(built.inline_shapes)} images", flush=True)
