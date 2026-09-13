"""Render the manuscript markdown into the SoftwareX article template.

Handles the subset of markdown the manuscript uses: headings, paragraphs,
pipe tables, figure references written as ``![caption](path)``, bullet lists
and inline bold/italic/code.  Superscript affiliation markers written as
``^a^`` are rendered as superscript runs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

SOURCE = Path("manuscript/supplementary_material.md")
TARGET = Path("manuscript/supplementary_material.docx")
FIGURE_WIDTH = Inches(6.0)

INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|\^.+?\^|\[.+?\]\(.+?\))")


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
            run.font.size = Pt(9.5)
        elif piece.startswith("^") and piece.endswith("^"):
            run = paragraph.add_run(piece[1:-1])
            run.font.superscript = True
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True
        elif piece.startswith("["):
            match = re.fullmatch(r"\[(.*?)\]\((.*?)\)", piece, re.DOTALL)
            if match:
                label, url = match.groups()
                run = paragraph.add_run(label or url)
                run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
            else:
                paragraph.add_run(piece)
        else:
            paragraph.add_run(piece)


def add_table(document: Document, rows: list[str]) -> None:
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], [r for r in cells[1:] if not set("".join(r)) <= set("-: ")]
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(table.rows[0].cells, header):
        cell.paragraphs[0].clear()
        add_inline(cell.paragraphs[0], text)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in body:
        target = table.add_row().cells
        for cell, text in zip(target, row):
            cell.paragraphs[0].clear()
            add_inline(cell.paragraphs[0], text)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
    document.add_paragraph()


def build() -> Path:
    lines = SOURCE.read_text(encoding="utf-8").split("\n")
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()

        if not line.strip():
            index += 1
            continue

        if line.startswith("|"):
            block: list[str] = []
            while index < len(lines) and lines[index].startswith("|"):
                block.append(lines[index])
                index += 1
            add_table(document, block)
            continue

        if line.startswith("!["):
            caption, path = re.match(r"!\[(.*?)\]\((.*?)\)", line).groups()
            if Path(path).exists():
                document.add_picture(path, width=FIGURE_WIDTH)
                document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, caption)
            for run in paragraph.runs:
                run.font.size = Pt(9.5)
            index += 1
            continue

        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            if level == 1:
                heading = document.add_heading(level=0)
                add_inline(heading, text)
            else:
                heading = document.add_heading(level=min(level - 1, 4))
                add_inline(heading, text)
            index += 1
            continue

        if line.lstrip().startswith(("- ", "* ")):
            while index < len(lines) and lines[index].lstrip().startswith(("- ", "* ")):
                paragraph = document.add_paragraph(style="List Bullet")
                add_inline(paragraph, lines[index].lstrip()[2:].strip())
                index += 1
            continue

        # a paragraph: join wrapped lines until a blank line or a block start
        buffer: list[str] = []
        while index < len(lines):
            current = lines[index].rstrip()
            if not current.strip() or current.startswith(("#", "|", "![")) or \
                    current.lstrip().startswith(("- ", "* ")):
                break
            buffer.append(current.strip())
            index += 1
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_inline(paragraph, " ".join(buffer))

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    document.save(TARGET)
    return TARGET


if __name__ == "__main__":
    path = build()
    words = len(SOURCE.read_text(encoding="utf-8").split())
    print(f"{path} written; source markdown holds {words} words", flush=True)
