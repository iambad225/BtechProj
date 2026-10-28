"""Render the stage 3 speaker report (Markdown, nested bullets) to PDF.

Usage (from the repo root; needs reportlab and the Windows Arial / Consolas / Segoe UI Symbol fonts):
    python _DOCS/source/build_speaker_report.py [report.md] [out.pdf]
Handles the subset of Markdown the report uses: #/## headings, --- rules, "- " bullets nested by
two spaces, **bold**, *italic* and `code`.
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, CondPageBreak, Frame, PageTemplate, Paragraph, Spacer
from reportlab.platypus.flowables import HRFlowable

HERE = Path(__file__).resolve().parent
DEFAULT_MD = HERE.parent / "project stage 3 presentation" / "mid_sem_speaker_report.md"

# ---------------- fonts ----------------
FD = "C:/Windows/Fonts"
for name, f in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"),
                ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf"),
                ("Consolas", "consola.ttf"), ("Consolas-Bold", "consolab.ttf"),
                ("SegoeSym", "seguisym.ttf")]:
    pdfmetrics.registerFont(TTFont(name, f"{FD}/{f}"))
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                   boldItalic="Arial-BoldItalic")
registerFontFamily("Consolas", normal="Consolas", bold="Consolas-Bold", italic="Consolas",
                   boldItalic="Consolas-Bold")
registerFontFamily("SegoeSym", normal="SegoeSym", bold="SegoeSym", italic="SegoeSym",
                   boldItalic="SegoeSym")
# glyphs Arial lacks; drawn from Segoe UI Symbol instead
SYMBOLS = set("\u207b\u2081\u2082\u21d2\u226a\u26a0\u2714\U0001f4a1\U0001f6e0")

# ---------------- palette / styles (same look as the methodology PDF) ----------------
TEAL = colors.HexColor("#1f6f78")
INK = colors.HexColor("#1f2a33")
BODY = colors.HexColor("#27313a")
MUTED = colors.HexColor("#5b6770")
RULE = colors.HexColor("#c9d6d9")
SAY = "#1e7a46"

title_st = ParagraphStyle("t", fontName="Arial-Bold", fontSize=19, leading=23, textColor=INK,
                          spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=13, leading=16.5, textColor=TEAL,
                    spaceBefore=10, spaceAfter=3, keepWithNext=1)
body = ParagraphStyle("b", fontName="Arial", fontSize=9.2, leading=12.4, textColor=BODY)
LEVEL_BULLETS = ["\u2022", "\u2013", "\u2022", "\u2013", "\u2022"]


def bullet_style(level):
    left = 11 + level * 13
    return ParagraphStyle(f"li{level}", parent=body, leftIndent=left, bulletIndent=left - 9,
                          spaceBefore=2.2 if level == 0 else 0.6,
                          fontSize=9.2 if level < 2 else 8.9, leading=12.2 if level < 2 else 11.8,
                          bulletFontName="Arial", bulletFontSize=8.5,
                          bulletColor=TEAL if level == 0 else MUTED)


def inline(text):
    """Markdown inline -> reportlab mini-HTML."""
    parts = text.split("`")
    out = []
    for i, part in enumerate(parts):
        part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if i % 2:                                              # inside `code`
            out.append(f'<font name="Consolas" size="8.4" color="#3b4a55">{part}</font>')
            continue
        part = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", part)
        part = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", part)
        out.append(part)
    s = "".join(out)
    return "".join(f'<font name="SegoeSym">{c}</font>' if c in SYMBOLS else c for c in s)


def decorate(text):
    """Colour the speaking-script labels and the quoted slide lines."""
    if text.startswith('**"'):                                 # a line as it appears on the slide
        return f'<font color="#1f6f78">{inline(text)}</font>'
    m = re.match(r"^(Say(?: \([^)]*\))?:)\s*(.*)$", text)
    if m:
        return f'<font color="{SAY}"><b>{m.group(1)}</b></font> {inline(m.group(2))}'
    m = re.match(r"^(Q:|A(?: \([^)]*\))?:)\s*(.*)$", text)
    if m:
        return f"<b>{m.group(1)}</b> {inline(m.group(2))}"
    return inline(text)


def build(md_path, out_path):
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "Speaker report")
    story = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:].strip()), title_st))
        elif line.startswith("## "):
            story += [CondPageBreak(40 * mm), Paragraph(inline(line[3:].strip()), h2),
                      HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=0,
                                 spaceAfter=3)]
        elif line.strip() == "---":
            story.append(Spacer(1, 4))
        else:
            m = re.match(r"^( *)- (.*)$", line)
            if m:
                level = min(len(m.group(1)) // 2, len(LEVEL_BULLETS) - 1)
                story.append(Paragraph(decorate(m.group(2)), bullet_style(level),
                                       bulletText=LEVEL_BULLETS[level]))
            else:
                story.append(Paragraph(inline(line.strip()), body))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.setFont("Arial", 7.6); canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9 * mm, title)
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(str(out_path), pagesize=A4, title=title, author="biopipe",
                          leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                          bottomMargin=18 * mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])
    doc.build(story)


if __name__ == "__main__":
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else md.with_suffix(".pdf")
    build(md, out)
    print(f"built {out}")
