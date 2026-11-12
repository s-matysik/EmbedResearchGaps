"""Render the response letter (markdown) into a Word document in the house format.

Arial 11 pt throughout, headings 11 pt with bold from the style, single line
spacing, code spans in Consolas 9 pt.  Reviewer quotations are italic, the
label that introduces them bold, as written in the markdown.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

sys.path.insert(0, ".")
from build_docx import complete_run_fonts  # noqa: E402

BODY_FONT, SIZE, CODE_SIZE = "Arial", 11, 9
INLINE = re.compile(r"(\*\*.*?\*\*|\*[^*]+?\*|`[^`]+?`)")


def add_inline(paragraph, text: str) -> None:
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**"):
            run = paragraph.add_run(piece[2:-2]); run.bold = True
        elif piece.startswith("*"):
            run = paragraph.add_run(piece[1:-1]); run.italic = True
        elif piece.startswith("`"):
            run = paragraph.add_run(piece[1:-1]); run.font.name = "Consolas"; run.font.size = Pt(CODE_SIZE)
            continue
        else:
            run = paragraph.add_run(piece)
        run.font.name = BODY_FONT


def build(source: str = "rev/response.md", target: str = "rev/Response_to_reviewers.docx") -> str:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(SIZE)
    for section in document.sections:
        section.left_margin = section.right_margin = Inches(1.0)
    for line in Path(source).read_text(encoding="utf-8").split("\n"):
        text = line.rstrip()
        if not text.strip() or text.strip() == "---":
            continue
        if text.startswith("## ") or text.startswith("### "):
            level = 1 if text.startswith("## ") else 2
            paragraph = document.add_heading(text.lstrip("# "), level)
            for run in paragraph.runs:
                run.font.name = BODY_FONT
                run.font.size = Pt(SIZE)
                run.font.color.rgb = None
            paragraph.paragraph_format.space_before = Pt(6)
        else:
            paragraph = document.add_paragraph()
            add_inline(paragraph, text)
        paragraph.paragraph_format.line_spacing = 1.0
    complete_run_fonts(document)
    document.save(target)
    return target


if __name__ == "__main__":
    print(build(*sys.argv[1:3]))
