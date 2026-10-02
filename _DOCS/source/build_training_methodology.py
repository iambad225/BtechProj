"""Build ERD-ERS_MI_ErrP_Training_Methodology.pdf (revision 2: two-model paradigm).

Usage (from the repo root; needs reportlab and the Windows Arial fonts):
    python _DOCS/source/build_training_methodology.py _DOCS/ERD-ERS_MI_ErrP_Training_Methodology.pdf
"""
import math
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String

OUT = sys.argv[1] if len(sys.argv) > 1 else "methodology_v2.pdf"

# ---------------- fonts ----------------
FD = "C:/Windows/Fonts"
for name, f in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"),
                ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")]:
    pdfmetrics.registerFont(TTFont(name, f"{FD}/{f}"))
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                   boldItalic="Arial-BoldItalic")

# ---------------- palette ----------------
TEAL = colors.HexColor("#1f6f78")
INK = colors.HexColor("#1f2a33")
BODY = colors.HexColor("#27313a")
MUTED = colors.HexColor("#5b6770")
RULE = colors.HexColor("#c9d6d9")
STRIPE = colors.HexColor("#f1f6f7")
CALLOUT = colors.HexColor("#e9f3f4")
# cue colours as shown on the app's Cue display
READY = colors.HexColor("#f1c40f")
READY_LT = colors.HexColor("#f8e38c")
REST = colors.HexColor("#4a90d9")
GO = colors.HexColor("#2e9e48")
WRONG = colors.HexColor("#d64541")
FIX = colors.HexColor("#9aa5ad")
FIX_LT = colors.HexColor("#d9dee2")

# ---------------- styles ----------------
title_st = ParagraphStyle("t", fontName="Arial-Bold", fontSize=21, leading=25, textColor=INK)
sub_st = ParagraphStyle("s", fontName="Arial", fontSize=10.5, leading=14, textColor=MUTED,
                        spaceBefore=3)
meta_st = ParagraphStyle("m", fontName="Arial", fontSize=8.3, leading=11.5, textColor=MUTED)
h1 = ParagraphStyle("h1", fontName="Arial-Bold", fontSize=13.5, leading=17, textColor=TEAL,
                    spaceBefore=13, spaceAfter=5, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=10.3, leading=13.5, textColor=INK,
                    spaceBefore=8, spaceAfter=3, keepWithNext=1)
body = ParagraphStyle("b", fontName="Arial", fontSize=9.1, leading=13, textColor=BODY,
                      spaceAfter=4)
bul = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
num = ParagraphStyle("nm", parent=body, leftIndent=14, bulletIndent=0, spaceAfter=2.5)
cell = ParagraphStyle("c", fontName="Arial", fontSize=8, leading=10.3, textColor=BODY)
cellb = ParagraphStyle("cb", parent=cell, fontName="Arial-Bold")
cellh = ParagraphStyle("ch", parent=cell, fontName="Arial-Bold", textColor=colors.white)
cap = ParagraphStyle("cap", fontName="Arial-Italic", fontSize=7.9, leading=10.5, textColor=MUTED,
                     spaceBefore=3, spaceAfter=8)
tcap = ParagraphStyle("tcap", fontName="Arial-Bold", fontSize=8.3, leading=11, textColor=MUTED,
                      spaceBefore=4, spaceAfter=3, keepWithNext=1)
ref_st = ParagraphStyle("r", fontName="Arial", fontSize=7.9, leading=10.6, textColor=BODY,
                        leftIndent=16, firstLineIndent=-16, spaceAfter=2)

W_CONTENT = A4[0] - 36 * mm


def P(text, st=body):
    return Paragraph(text, st)


def bullets(items, st=bul):
    return [Paragraph(t, st, bulletText="\u2022") for t in items]


def steps(items):
    return [Paragraph(t, num, bulletText=f"{i}.") for i, t in enumerate(items, 1)]


def table(header, rows, widths, group_col=None, bold_col=None):
    """Table with a teal header. ``group_col`` spans + shades consecutive equal values."""
    data = [[Paragraph(h, cellh) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), cellb if (bold_col is not None and j == bold_col) else cell)
                     for j, c in enumerate(r)])
    st = [("BACKGROUND", (0, 0), (-1, 0), TEAL),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BOX", (0, 0), (-1, -1), 0.5, RULE),
          ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if group_col is None:
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), STRIPE))
    else:
        start, shade = 1, False
        for i in range(2, len(data) + 1):
            end_group = i == len(data) or rows[i - 1][group_col] != rows[start - 1][group_col]
            if end_group:
                if shade:
                    st.append(("BACKGROUND", (0, start), (-1, i - 1), STRIPE))
                if i - 1 > start:
                    st.append(("SPAN", (group_col, start), (group_col, i - 1)))
                    for k in range(start + 1, i):
                        data[k][group_col] = Paragraph("", cell)
                data[start][group_col] = Paragraph(str(rows[start - 1][group_col]), cellb)
                start, shade = i, not shade
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(st))
    t.spaceAfter = 8
    return t


def callout(flows):
    t = Table([[flows]], colWidths=[W_CONTENT])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CALLOUT),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def checkbox():
    d = Drawing(9, 9)
    d.add(Rect(0, 0, 8.5, 8.5, fillColor=colors.white, strokeColor=TEAL, strokeWidth=0.8))
    return d


# ---------------- drawing helpers ----------------
def _arrow(d, x1, y1, x2, y2, color=INK, w=0.9):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w))
    a = math.atan2(y2 - y1, x2 - x1)
    L, s = 5.5, 0.42
    p1 = (x2 - L * math.cos(a - s), y2 - L * math.sin(a - s))
    p2 = (x2 - L * math.cos(a + s), y2 - L * math.sin(a + s))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=color, strokeColor=color,
                  strokeWidth=0.4))


def _box(d, x, y, w, h, title, subs=(), fill=colors.white, stroke=TEAL, txt=INK, dash=None):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.9,
               strokeDashArray=dash))
    lines = [(title, "Arial-Bold", 8.4)] + [(s, "Arial", 7.3) for s in subs]
    total = 10 + 9 * (len(lines) - 1)
    yy = y + h / 2 + total / 2 - 8
    for t, fn, fs in lines:
        d.add(String(x + w / 2, yy, t, fontName=fn, fontSize=fs, fillColor=txt,
                     textAnchor="middle"))
        yy -= 9.5


def _label(d, x, y, t, size=7, color=MUTED, anchor="start", font="Arial-Italic"):
    d.add(String(x, y, t, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def fig_control_loop():
    W, H = 480, 160
    d = Drawing(W, H)
    bw, gap = 96, 32
    xs = [0, bw + gap, 2 * (bw + gap), 3 * (bw + gap)]
    y1, h1_ = 92, 36
    y2, h2_ = 22, 42
    # system-wide EMG gate (planned)
    d.add(Rect(0, 140, W, 18, rx=4, ry=4, fillColor=colors.white, strokeColor=FIX,
               strokeWidth=0.8, strokeDashArray=[3, 2]))
    d.add(String(W / 2, 146, "EMG wrist-clench: system-wide on/off gate (planned, not built yet)",
                 fontName="Arial-Italic", fontSize=7.6, fillColor=MUTED, textAnchor="middle"))
    _box(d, xs[0], y1, bw, h1_, "EEG stream", ["8 ch \u00b7 250 Hz"])
    _box(d, xs[1], y1, bw, h1_, "MI model", ["rest \u00b7 left \u00b7 right"], fill=TEAL,
         txt=colors.white)
    _box(d, xs[2], y1, bw, h1_, "left / right", ["\u2192 command"])
    _box(d, xs[3], y1, bw, h1_, "Drone action", ["(after integration)"])
    for i in range(3):
        _arrow(d, xs[i] + bw, y1 + h1_ / 2, xs[i + 1] - 1, y1 + h1_ / 2)
    _box(d, xs[1], y2, bw, h2_, "rest", ["gate closed", "no command"], fill=FIX_LT, stroke=FIX)
    _box(d, xs[2], y2, bw, h2_, "keep / undo", ["correct \u2192 keep", "error \u2192 veto"])
    _box(d, xs[3], y2, bw, h2_, "ErrP model", ["correct \u00b7 error", "on the feedback"],
         fill=TEAL, txt=colors.white)
    _arrow(d, xs[1] + bw / 2, y1, xs[1] + bw / 2, y2 + h2_ + 1)
    _label(d, xs[1] + bw / 2 + 4, y1 - 15, "if rest")
    _arrow(d, xs[3] + bw / 2, y1, xs[3] + bw / 2, y2 + h2_ + 1)
    _label(d, xs[3] + bw / 2 + 4, y1 - 15, "outcome shown")
    _arrow(d, xs[3], y2 + h2_ / 2, xs[2] + bw + 1, y2 + h2_ / 2)
    _label(d, (xs[1] + bw + xs[2]) / 2, y1 + h1_ + 3, "if left/right", anchor="middle")
    return d


def fig_mi_timeline():
    W, H = 480, 98
    d = Drawing(W, H)
    x0, pps = 15, 25.0
    X = lambda t: x0 + t * pps
    yb, hb = 42, 24
    segs = [(0, 2, "ready", None), (2, 6, "REST", "rest"), (6, 8, "ready", None),
            (8, 12, "LEFT", "left"), (12, 14, "ready", None), (14, 18, "RIGHT", "right")]
    for s, e, lab, cls in segs:
        if lab == "ready":
            d.add(Rect(X(s), yb, X(e) - X(s), hb, fillColor=READY_LT, strokeColor=colors.white,
                       strokeWidth=1))
            d.add(String((X(s) + X(e)) / 2, yb + 9, "get ready", fontName="Arial", fontSize=7,
                         fillColor=INK, textAnchor="middle"))
        else:
            fill = REST if lab == "REST" else GO
            d.add(Rect(X(s), yb, X(e) - X(s), hb, fillColor=fill, strokeColor=colors.white,
                       strokeWidth=1))
            d.add(String((X(s) + X(e)) / 2, yb + 8.5, f"{lab}  \u00b7  4 s", fontName="Arial-Bold",
                         fontSize=8.3, fillColor=colors.white, textAnchor="middle"))
            d.add(Rect(X(s), yb, 0.4 * pps, hb, fillColor=colors.Color(0, 0, 0, alpha=0.28),
                       strokeColor=None))
            # trial bracket
            yk = yb + hb + 6
            d.add(Line(X(s) + 1, yk, X(e) - 1, yk, strokeColor=MUTED, strokeWidth=0.7))
            d.add(Line(X(s) + 1, yk, X(s) + 1, yk - 3, strokeColor=MUTED, strokeWidth=0.7))
            d.add(Line(X(e) - 1, yk, X(e) - 1, yk - 3, strokeColor=MUTED, strokeWidth=0.7))
            _label(d, (X(s) + X(e)) / 2, yk + 3, f"1 trial ({cls})", anchor="middle")
    # axis
    d.add(Line(X(0), yb - 4, X(18), yb - 4, strokeColor=MUTED, strokeWidth=0.6))
    for t in range(0, 19, 2):
        d.add(Line(X(t), yb - 4, X(t), yb - 7, strokeColor=MUTED, strokeWidth=0.6))
        d.add(String(X(t), yb - 15, f"{t}" + (" s" if t == 18 else ""), fontName="Arial",
                     fontSize=6.8, fillColor=MUTED, textAnchor="middle"))
    # legend
    ly, lx = 4, X(0)
    for fill, txt in [(READY_LT, "get ready (2 s, not recorded)"),
                      (REST, "recorded block = 1 trial"),
                      (colors.Color(0, 0, 0, alpha=0.45), "0.4 s onset trim (dropped)")]:
        d.add(Rect(lx, ly, 9, 7, fillColor=fill, strokeColor=None))
        d.add(String(lx + 12, ly + 1, txt, fontName="Arial", fontSize=7, fillColor=BODY))
        lx += 12 + pdfmetrics.stringWidth(txt, "Arial", 7) + 16
    return d


def fig_errp_timeline():
    W, H = 480, 90
    d = Drawing(W, H)
    x0, pps = 15, 128.0
    X = lambda t: x0 + t * pps
    yb, hb = 44, 28
    d.add(Rect(X(0), yb, X(0.9) - X(0), hb, fillColor=FIX, strokeColor=colors.white))
    d.add(String((X(0) + X(0.9)) / 2, yb + 15, "fixation  +", fontName="Arial-Bold", fontSize=8,
                 fillColor=colors.white, textAnchor="middle"))
    d.add(String((X(0) + X(0.9)) / 2, yb + 5, "0.7\u20131.1 s (jittered)", fontName="Arial",
                 fontSize=6.8, fillColor=colors.white, textAnchor="middle"))
    d.add(Rect(X(0.9), yb, X(1.8) - X(0.9), hb, fillColor=REST, strokeColor=colors.white))
    d.add(String((X(0.9) + X(1.8)) / 2, yb + 15, "TARGET arrow", fontName="Arial-Bold",
                 fontSize=8, fillColor=colors.white, textAnchor="middle"))
    d.add(String((X(0.9) + X(1.8)) / 2, yb + 5, "0.9 s", fontName="Arial", fontSize=6.8,
                 fillColor=colors.white, textAnchor="middle"))
    d.add(Rect(X(1.8), yb + hb / 2, X(2.8) - X(1.8), hb / 2, fillColor=GO,
               strokeColor=colors.white))
    d.add(Rect(X(1.8), yb, X(2.8) - X(1.8), hb / 2, fillColor=WRONG, strokeColor=colors.white))
    d.add(String((X(1.8) + X(2.8)) / 2, yb + hb / 2 + 4, "OK (matches target)",
                 fontName="Arial-Bold", fontSize=7.4, fillColor=colors.white, textAnchor="middle"))
    d.add(String((X(1.8) + X(2.8)) / 2, yb + 4, "WRONG (at error rate)", fontName="Arial-Bold",
                 fontSize=7.4, fillColor=colors.white, textAnchor="middle"))
    d.add(Rect(X(2.8), yb, X(3.45) - X(2.8), hb, fillColor=FIX_LT, strokeColor=colors.white))
    d.add(String((X(2.8) + X(3.45)) / 2, yb + 10, "next trial", fontName="Arial", fontSize=7,
                 fillColor=INK, textAnchor="middle"))
    _label(d, (X(1.8) + X(2.8)) / 2, yb + hb + 4, "outcome · 1.0 s", anchor="middle")
    # event marker (label to the left of the line, clear of the outcome label)
    d.add(Line(X(1.8), yb - 14, X(1.8), yb + hb + 16, strokeColor=INK, strokeWidth=1.1))
    d.add(String(X(1.8) - 4, yb + hb + 9, "outcome onset = event, time-stamped correct / error",
                 fontName="Arial-Bold", fontSize=7.2, fillColor=INK, textAnchor="end"))
    # epoch
    ye, he = 20, 11
    d.add(Rect(X(1.6), ye, X(2.6) - X(1.6), he, fillColor=CALLOUT, strokeColor=TEAL,
               strokeWidth=0.8))
    d.add(Rect(X(1.6), ye, X(1.8) - X(1.6), he, fillColor=colors.HexColor("#b9d3d6"),
               strokeColor=TEAL, strokeWidth=0.8))
    for t, lab in [(1.6, "\u22120.2"), (1.8, "0"), (2.6, "+0.8 s")]:
        d.add(String(X(t), ye - 9, lab, fontName="Arial", fontSize=6.8, fillColor=MUTED,
                     textAnchor="middle"))
    d.add(String(X(1.6) - 6, ye + 2.5, "epoch used by the ErrP model (shaded = baseline)",
                 fontName="Arial", fontSize=7, fillColor=BODY, textAnchor="end"))
    return d


# ---------------- page furniture ----------------
def on_page(c, doc):
    c.saveState()
    c.setStrokeColor(RULE); c.setLineWidth(0.5)
    c.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    c.setFont("Arial", 7.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "biopipe \u2014 ERD/ERS, MI & ErrP training methodology "
                                   "\u00b7 revision 2")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    c.restoreState()


# ---------------- content ----------------
S = []
S += [P("ERD/ERS, MI &amp; ErrP \u2014 Training Methodology", title_st),
      P("Two-model EEG paradigm: 3-class motor imagery (rest / left / right) with the ERD/ERS "
        "gate built in, plus error-related potentials", sub_st),
      Spacer(1, 6), HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=6),
      P("biopipe / Hybrid EEG\u2013EMG BMI \u2022 OpenBCI Cyton (8 ch, 250 Hz) + Ultracortex "
        "\u2022 Montage Fz, FC5, FC6, C3, Cz, C4, P3, P4 (A1 ref, A2 bias) \u2022 Revision 2, "
        "2 October 2026 \u2014 replaces the 21 September 2026 three-model version (changes listed "
        "at the end)", meta_st)]

# 1 Overview
S += [P("1. Overview", h1),
      P("The EEG side trains exactly <b>two models</b>:"),
      *bullets([
          "<b>MI model \u2014 rest / mi_left / mi_right.</b> One 3-class classifier. Its <i>rest</i> "
          "class is the ERD/ERS gate: sensorimotor mu/beta rhythms are strong at rest and "
          "desynchronise during imagery, so \u201crest vs imagery\u201d is the gate and "
          "\u201cleft vs right\u201d is the command \u2014 learned together, with no separate gate model.",
          "<b>ErrP model \u2014 correct / error.</b> Detects the brain\u2019s response to a wrong "
          "outcome so the controller can veto or undo an action."]),
      Spacer(1, 4), fig_control_loop(),
      P("Figure 1 \u2014 Control loop. A <i>rest</i> prediction closes the gate; a left/right "
        "prediction becomes a command, and the ErrP model checks the brain\u2019s response to its "
        "outcome.", cap)]

# 2 Montage (kept together so the table never strands its heading)
S += [KeepTogether([
      P("2. Montage", h1),
      P("Physically available holes only (FCz, FC3/FC4, CP3/CP4 and CPz are unavailable). Load it "
        "with <b>Config \u2192 Load MI/ErrP preset</b>. Standard OpenBCI EEG wiring: electrodes in "
        "the bottom-row inputs N1P\u2013N8P, reference ear clip in SRB2, bias ear clip in BIAS."),
      P("Table 1 \u2014 Electrode map", tcap),
      table(["Electrode", "GUI ch", "Cyton input", "Used for"],
            [["Fz", "0", "N1P (ch 1)", "ErrP fronto-central (ERN/FRN); blink check"],
             ["FC5", "1", "N2P (ch 2)", "MI \u2014 left premotor/hand area (CSP)"],
             ["FC6", "2", "N3P (ch 3)", "MI \u2014 right premotor/hand area (CSP)"],
             ["C3", "3", "N4P (ch 4)", "MI \u2014 left hand area: ERD during <i>right</i>-hand imagery"],
             ["Cz", "4", "N5P (ch 5)", "MI midline; ErrP central (ERN, Pe)"],
             ["C4", "5", "N6P (ch 6)", "MI \u2014 right hand area: ERD during <i>left</i>-hand imagery"],
             ["P3", "6", "N7P (ch 7)", "MI posterior; ErrP Pe; eyes-closed alpha check"],
             ["P4", "7", "N8P (ch 8)", "MI posterior; ErrP Pe; eyes-closed alpha check"],
             ["A1 (left ear)", "ref", "SRB2", "Reference \u2014 required"],
             ["A2 (right ear)", "bias", "BIAS", "Bias (driven ground) \u2014 required"]],
            [72, 42, 70, W_CONTENT - 184], bold_col=0)])]

# 3 Shared settings
S += [KeepTogether([P("3. Shared EEG settings (Config and Monitor tabs)", h1),
      P("These apply to both models. Notch, band limits and common-average reference change only "
        "what the Monitor shows: recordings are saved <b>raw</b>, and each model applies its own "
        "band-pass when it trains and runs (Appendix A)."),
      P("Table 2 \u2014 Config and Monitor settings", tcap),
      table(["Tab", "Setting", "Recommended", "Options / range", "Affects"],
            [["Config", "Profile", "One per setup, e.g. <i>MI-8ch</i>", "Load \u00b7 Delete \u00b7 Save as",
              "Restores every Config setting below in one click."],
             ["Config", "Source", "cyton (synthetic for a dry run)", "cyton \u00b7 lsl \u00b7 synthetic",
              "Where samples come from."],
             ["Config", "Cyton port", "The dongle\u2019s port", "e.g. COM3",
              "Device Manager \u2192 Ports. Close the OpenBCI GUI first."],
             ["Config", "Sample rate", "250", "Hz", "Must match the board (Cyton: 250 Hz)."],
             ["Config", "Montage (10-20 head)", "Load MI/ErrP preset",
              "Click sets Data \u00b7 Ref (SRB) \u00b7 Bias \u00b7 Clear; channel 0\u20137 per Data node",
              "Electrode\u2192channel map saved with every recording. Live and pooling need the same map."],
             ["Config", "Actions", "mi_left, mi_right", "Add \u00b7 Remove",
              "The two MI classes (rest is added automatically). Names containing "
              "\u201cleft\u201d/\u201cright\u201d set the cue arrows. Keep exactly these two."],
             ["Config", "Notch Hz", "50", "Hz", "Monitor only (India mains)."],
             ["Config", "Band low / high", "1 / 40 (high 45\u201350 to see \u03b3)", "Hz", "Monitor only."],
             ["Config", "Common-avg ref", "On", "On \u00b7 off", "Monitor only."],
             ["Config", "Window ms", "1000\u20132000", "ms", "Monitor band-power window."],
             ["Config", "Step ms", "250", "ms (Live minimum 150)", "Monitor refresh and Live update rate."],
             ["Config", "Alpha thresh", "\u2248 0.2, then calibrate", "0\u20131 (alpha share)",
              "Alpha light; electrodes set to the most posterior pair (P3/P4) on Apply."],
             ["Config", "Apply &amp; Save", "After any change", "\u2014",
              "Writes config/eeg.yaml and stops the stream \u2014 press Start again."],
             ["Monitor", "alpha test", "On", "Checkbox", "Lights when P3/P4 alpha share exceeds Alpha thresh."],
             ["Monitor", "waterfall", "On during checks", "Checkbox",
              "Spectrogram: frequency across (0 \u2192 Band high), time downward."],
             ["Monitor", "simulate eyes-closed", "\u2014", "Synthetic source only",
              "Injects alpha to test the pipeline without hardware."]],
            [44, 74, 92, 98, W_CONTENT - 308], group_col=0)])]

# 4 Setup
S += [P("4. Headset setup and signal quality (every session)", h1),
      *bullets([
          "<b>Connect reference and bias.</b> A1 \u2192 SRB2, A2 \u2192 BIAS (never AVDD/AVSS/AGND). "
          "Without both you record mains hum and drift, not EEG \u2014 the giveaway is a band "
          "composition swamped by \u03b4.",
          "<b>Contact.</b> Part the hair and twist each comb down to the scalp. Aim for impedances in "
          "the low tens of k\u03a9 and similar across channels; mismatched or very high contacts let "
          "mains leak in, and common-average reference spreads one bad channel into all of them.",
          "<b>Monitor checks</b> (EEG \u2192 Monitor \u2192 Start): no channel flagged <b>RAILED</b>, "
          "<b>FLAT</b> or <i>near rail</i> \u2014 judge by the flags, not the traces: with common-average "
          "reference on, a dead channel still draws a convincing trace. A blink gives a "
          "slow deflection at Fz; a jaw clench gives a broadband burst on all channels that settles "
          "back; the band composition shows a spread of \u03b8/\u03b1/\u03b2 rather than a wall of \u03b4.",
          "<b>Eyes-closed test.</b> 15 s eyes open, 15 s closed: alpha should rise at P3/P4 (alpha "
          "light, or a bright ~10 Hz column on the waterfall). Parietal alpha is weaker than "
          "occipital, so set Alpha thresh between your eyes-open and eyes-closed values.",
          "<b>Environment.</b> Upright and relaxed, screen at eye level ~60 cm; unplug the laptop "
          "charger if possible; away from mains cables and motors; phone away; no gum."])]

# 5 Rules
S += [P("5. Recording rules (both models)", h1),
      *bullets([
          "Stay still and silent during recorded periods \u2014 no talking, head movement or shifting.",
          "Blink only during the yellow <i>get ready</i> cue (MI) or the fixation cross (ErrP) \u2014 "
          "never during REST, imagery or an outcome.",
          "No jaw clenching or facial/neck tension: muscle activity swamps EEG.",
          "Keep a soft, steady gaze on the cue. Ignore the small countdown in the corner \u2014 it is "
          "deliberately dim and off-centre.",
          "Runs of about 3\u20135 minutes with breaks; fatigue degrades both signals."])]

# 6 MI
S += [P("6. Model 1 \u2014 Motor imagery (rest / left / right)", h1),
      P("6.1 Why rest is a class", h2),
      P("Imagining a hand movement desynchronises the mu (8\u201312 Hz) and beta (13\u201330 Hz) "
        "rhythms over the opposite hemisphere\u2019s hand area \u2014 C3 for the right hand, C4 for the "
        "left \u2014 and they rebound (ERS) afterwards [1, 2]. During rest they stay high. Training "
        "<i>rest</i> as its own class therefore teaches the ERD/ERS gate and the left/right decision "
        "in one model; P(rest) acts as a soft gate."),
      P("6.2 Subject technique", h2),
      *bullets([
          "<b>Imagine, don\u2019t move.</b> Imagine the <i>feeling</i> of slowly squeezing the cued "
          "hand, as if your own muscles did it \u2014 not a video of yourself. Hands stay relaxed and "
          "still. A real squeeze is motor execution (another signal plus muscle artefact), so a model "
          "trained on it fails on pure imagery.",
          "<b>Warm-up (not recorded):</b> 3\u20135 slow real squeezes per hand, then reproduce that "
          "feeling without moving.",
          "Use the <b>same strategy every trial</b> \u2014 consistency beats intensity.",
          "<b>Yellow</b> = prepare, don\u2019t start. <b>Green</b> = imagine for the whole block; stop "
          "when the cue changes. <b>Blue REST</b> = relax, fixate and keep still \u2014 rest is "
          "recorded data too."]),
      P("6.3 Recording protocol", h2),
      P("Each rep records REST, LEFT and RIGHT once each <b>in random order</b>, each after an "
        "unrecorded 2 s <i>get ready</i> cue; counts stay equal and no cue repeats back-to-back. Each "
        "recorded block is one <b>trial</b>; its first 0.4 s is dropped as cue-onset transient. A rep "
        "takes 18 s at 4 s per class, so 10 reps is a 3-minute run."),
      fig_mi_timeline(),
      P("Figure 2 \u2014 One example MI rep at the default 4 s per class (colours as on the Cue "
        "display). The order inside each rep is randomised.", cap),
      P("<b>How much:</b> aim for at least 25\u201330 trials per class \u2014 for example 3 runs of "
        "10 reps. Cross-validation uses up to 5 folds; with fewer than 5 trials in a class it uses "
        "fewer folds, and with fewer than 2 it cannot validate at all."),
      P("6.4 Features and model", h2),
      P("Raw window \u2192 band-pass 8\u201330 Hz \u2192 Common Spatial Patterns (CSP) \u2192 log-variance "
        "\u2192 standardise \u2192 classifier. CSP learns spatial filters that maximise the variance "
        "difference between classes [3]. With three classes it runs one-vs-rest \u2014 a CSP per class "
        "against the other two, 4 filters each \u2014 giving 12 features [4]."),
      P("Table 3 \u2014 Classifier choices (Train \u2192 Model)", tcap),
      table(["Classifier", "Type", "When to use"],
            [["lda", "Linear discriminant (default)",
              "Start here: robust with little data, fast; the standard MI choice [5]."],
             ["svm", "Linear support-vector machine", "Often matches LDA; may edge ahead with more trials."],
             ["logreg", "Logistic regression (linear)", "Similar boundary to LDA; well-behaved probabilities."],
             ["rf", "Random forest, 200 trees", "Non-linear; needs more data to beat the linear models."],
             ["tree", "Single decision tree",
              "Interpretable but tends to overfit CSP features \u2014 use for comparison only."]],
            [56, 140, W_CONTENT - 196], bold_col=0),
      P("<b>What to expect:</b> chance is 33%. With dry electrodes an early target is roughly "
        "50\u201370% balanced accuracy, with rest vs imagery usually separating more easily than left "
        "vs right. A sizeable minority of users need several sessions before their imagery "
        "patterns separate [6] \u2014 treat these figures as indicative, not a guarantee."),
      P("6.5 Settings", h2),
      P("Table 4 \u2014 MI settings in the GUI", tcap),
      table(["Tab", "Setting", "Recommended", "Options / range", "Notes"],
            [["Record", "Subject", "Your ID, e.g. <i>amogh</i>", "Free text",
              "Names the folder eeg_&lt;subject&gt;_&lt;date-time&gt;."],
             ["Record", "Paradigm", "MI (left/right)", "MI \u00b7 ErrP",
              "REST, LEFT and RIGHT blocks (3-class)."],
             ["Record", "Seconds/class", "4", "2\u201330 s",
              "Length of each recorded block; must exceed the Train window + 0.4 s."],
             ["Record", "Reps", "10 per run", "1\u201315",
              "Each class once per rep, random order."],
             ["Record", "Description", "Run notes", "Optional", "Saved in the recording\u2019s JSON."],
             ["Train", "Modality \u00b7 Paradigm", "EEG \u00b7 MI (rest/L/R)", "\u2014",
              "Lists only MI recordings."],
             ["Train", "Model", "lda", "lda \u00b7 svm \u00b7 logreg \u00b7 rf \u00b7 tree",
              "See Table 3."],
             ["Train", "Win / Step ms", "1000 / 250", "ms",
              "Training window and hop; also the model\u2019s Live window. Longer is steadier but "
              "slower to respond."],
             ["Train", "Recordings", "2\u20133 runs, pooled", "Ctrl/Shift-click; Refresh",
              "Same montage, rate and labels required."],
             ["Train", "Description", "Model notes", "Optional", "Saved in the model\u2019s JSON."],
             ["Live", "Models", "The saved MI model", "Multi-select; refresh; Start",
              "Window from the model; update rate from Config Step."]],
            [40, 76, 94, 96, W_CONTENT - 306], group_col=0),
      P("6.6 In the app", h2),
      *steps([
          "<b>Config</b> \u2192 Load MI/ErrP preset \u2192 Source <i>cyton</i> + COM port \u2192 Apply &amp; Save.",
          "<b>Monitor</b> \u2192 run the checks in section 4.",
          "<b>Record</b> \u2192 Subject \u2192 Paradigm <i>MI (left/right)</i> \u2192 Seconds/class 4, "
          "Reps 10 \u2192 Description \u2192 Record session, and follow the cue. Repeat for 3 runs "
          "with breaks.",
          "<b>Train</b> \u2192 EEG \u2192 <i>MI (rest/L/R)</i> \u2192 Model <i>lda</i> \u2192 Win/Step "
          "1000/250 \u2192 Ctrl-click the runs \u2192 Train \u2192 read the report \u2192 Save model.",
          "Optionally retrain the same runs with svm, logreg, rf or tree and keep the model with the "
          "best macro-F1."]),
]

# 7 ErrP
S += [P("7. Model 2 \u2014 Error-related potentials (correct / error)", h1),
      P("7.1 Subject technique", h2),
      *bullets([
          "<b>Care about the outcome.</b> Watch the target, then the result, and let the natural "
          "\u201cthat\u2019s wrong\u201d reaction happen.",
          "Never move, sigh or clench on an error \u2014 a movement time-locked to errors is the fastest "
          "way to ruin the data, because the model learns the artefact.",
          "Don\u2019t try to predict outcomes; react honestly. Blink during the fixation cross."]),
      P("7.2 Recording protocol", h2),
      P("A cued feedback task on the Cue display: fixation, a blue target arrow, then an outcome "
        "arrow. At the set error rate the system deliberately shows the other direction. The "
        "outcome onset is time-stamped as <i>correct</i> or <i>error</i>, and each event yields one "
        "epoch \u2014 one trial for cross-validation."),
      KeepTogether([fig_errp_timeline(),
                    P("Figure 3 \u2014 One ErrP trial and the epoch extracted around the outcome "
                      "onset.", cap)]),
      P("<b>How much:</b> about 3 s per trial, so 100 trials take roughly 5 minutes. Aim for at "
        "least 200 trials (about 60 errors at 0.30) over 2 runs; 60 trials give only about 18 errors, "
        "too few for a stable 5-fold estimate."),
      P("7.3 Features and model", h2),
      P("Band-pass 1–10 Hz keeps the slow error components — the error-related negativity at "
        "fronto-central sites, then the error positivity (Pe) [7, 8]. Each epoch is baseline-corrected "
        "on −0.2 to 0 s, split into 12 time bins and averaged per bin and channel (12 × 8 = 96 "
        "features), then classified by shrinkage LDA with balanced class priors — the standard "
        "single-trial ERP classifier [9]."),
      P("<b>What to expect:</b> about 65–80% balanced accuracy in good setups. Outcomes are "
        "time-stamped in software, so tens of milliseconds of display jitter blur the epochs; be "
        "sceptical of high scores from few trials."),
      P("7.4 Settings", h2),
      P("Table 5 \u2014 ErrP settings in the GUI", tcap),
      table(["Tab", "Setting", "Recommended", "Options / range", "Notes"],
            [["Record", "Subject", "Your ID", "Free text", "As for MI."],
             ["Record", "Paradigm", "ErrP (feedback task)", "MI \u00b7 ErrP",
              "Runs the target/outcome task on the Cue display."],
             ["Record", "Trials", "100 per run", "20\u2013400", "About 3 s per trial."],
             ["Record", "Error rate", "0.30", "0.10\u20130.50 (step 0.05)",
              "Share of deliberately wrong outcomes. 0.20\u20130.30: rarer errors evoke a stronger "
              "response but give fewer error epochs."],
             ["Record", "Description", "Run notes", "Optional", "Saved in the recording\u2019s JSON."],
             ["Train", "Modality \u00b7 Paradigm", "EEG \u00b7 ErrP", "\u2014", "Lists only ErrP recordings."],
             ["Train", "Model", "\u2014 (hidden)", "Fixed", "Always shrinkage LDA."],
             ["Train", "Win / Step ms", "\u2014 (ignored)", "\u2014",
              "ErrP uses fixed \u22120.2 to +0.8 s epochs."],
             ["Train", "Recordings", "2 or more runs, pooled", "Ctrl/Shift-click; Refresh",
              "Same pooling rule as MI."],
             ["Live", "\u2014", "Not run continuously", "\u2014",
              "ErrP is event-locked; it will be evaluated after each action once the control loop "
              "is integrated."]],
            [40, 76, 94, 96, W_CONTENT - 306], group_col=0),
      P("7.5 In the app", h2),
      *steps([
          "Config preset and Monitor checks as for MI.",
          "<b>Record</b> \u2192 Subject \u2192 Paradigm <i>ErrP (feedback task)</i> \u2192 Trials 100, "
          "Error rate 0.30 \u2192 Record session, and watch the screen. Repeat for a second run.",
          "<b>Train</b> \u2192 EEG \u2192 <i>ErrP</i> \u2192 Ctrl-click the runs \u2192 Train \u2192 Save model."]),
]

# 8 Evaluation
S += [P("8. Evaluation: how to read the report", h1),
      P("Training never splits windows at random. Neighbouring windows from the same block are "
        "nearly identical, so a random split leaks test data into training and inflates the score "
        "[10]. Instead:"),
      *bullets([
          "<b>Grouping:</b> every MI block and every ErrP event is one trial, and all of a "
          "trial\u2019s windows or epochs stay in the same fold.",
          "<b>Folds:</b> StratifiedGroupKFold with k = min(5, fewest trials in any class); classes "
          "are balanced across folds.",
          "<b>Report:</b> out-of-fold predictions are pooled into the per-class report shown in the "
          "Train tab\u2019s right-hand box; the saved model is then refit on all the data."]),
      P("Table 6 \u2014 Metrics in the report", tcap),
      table(["Metric", "What it means", "What to look for"],
            [["precision", "Correct share of a class\u2019s predictions.",
              "Low for left/right \u2192 rest misread as a command."],
             ["recall", "Share of a class\u2019s windows caught.",
              "MI rest: gate reliability. ErrP error: errors caught."],
             ["F1", "Harmonic mean of precision and recall.",
              "Compare classifiers by macro-F1 (mean over classes)."],
             ["support", "Windows (MI) or epochs (ErrP) per class.",
              "~11 per 4 s block; if uneven, record more of that class."],
             ["balanced accuracy \u00b1 std", "Mean per-class recall; mean \u00b1 std over folds.",
              "Chance 33% (MI) / 50% (ErrP); large std \u2192 record more."]],
            [92, 165, W_CONTENT - 257], bold_col=0),
      P("Within-session cross-validation is still optimistic. The honest test is a new session "
        "\u2014 re-seat the headset on another day and run the saved model in Live.")]

# 9 Live
S += [P("9. Live use", h1),
      *bullets([
          "EEG \u2192 Live \u2192 refresh \u2192 select the MI model \u2192 Start. Bars show P(rest), "
          "P(left) and P(right), with the top class highlighted and repeated on the DECISION line.",
          "<i>Rest</i> = gate closed, no command. Live currently reports the top class at every "
          "update; a confidence threshold and smoothing for drone commands will be tuned during "
          "integration.",
          "The current Config montage must contain the model\u2019s electrodes (the preset guarantees "
          "this). The window comes from the model and the update rate from Config Step "
          "(minimum 150 ms).",
          "ErrP models are skipped in Live because they are event-locked."])]

# 10 Cadence
S += [P("10. Session cadence and pooling", h1),
      *bullets([
          "Short runs with breaks; two sessions on different days beat one long session.",
          "Retrain each session: EEG shifts between sessions and headset placements.",
          "Pool runs only when montage, sample rate and labels match \u2014 the Train tab blocks "
          "mismatches. Trial IDs stay unique across pooled runs.",
          "Record MI and ErrP as separate sessions."])]

# 11 Checklist
chk_items = [
    "A1 \u2192 SRB2 and A2 \u2192 BIAS clipped on and wired.",
    "Combs on the scalp; contacts low and even.",
    "Config: MI/ErrP preset, Source cyton + COM port, Apply &amp; Save.",
    "Monitor: no RAILED/FLAT flags, blink visible at Fz, no wall of \u03b4, alpha rises at P3/P4 with eyes closed.",
    "Charger unplugged or away from mains; phone away; seated upright.",
    "Record: Subject entered, paradigm chosen, description added.",
    "MI: imagine — don’t squeeze; same strategy; still during REST; blink only on yellow.",
    "ErrP: care about the outcome; don\u2019t move on errors.",
]
chk = Table([[checkbox(), Paragraph(t, cell)] for t in chk_items], colWidths=[16, W_CONTENT - 16])
chk.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                         ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
S += [KeepTogether([P("11. Pre-flight checklist", h1), chk])]

# Changes from revision 1
S += [KeepTogether([
    P("Changes from revision 1", h1),
    callout(bullets([
        "ERD/ERS is no longer a separate model — it is the MI model’s <i>rest</i> class.",
        "MI is 3-class with a selectable classifier (lda, svm, logreg, rf, tree).",
        "MI cues are block-randomised: equal rest/left/right counts, never the same cue twice in a row.",
        "<i>Rest</i> blocks are recorded and trained, so they must be as clean as imagery blocks "
        "(no blinking or fidgeting).",
        "Evaluation is trial-grouped, class-stratified k-fold cross-validation with per-class "
        "precision, recall, F1 and support.",
        "Recordings carry a <i>trial</i> column; the subject is entered in the Record tab.",
        "Reference (SRB) and bias must be physically connected; the alpha check runs on P3/P4."]))])]

# Appendix
S += [P("Appendix A \u2014 Settings fixed in code (not in the GUI)", h1),
      P("Table 7 \u2014 Fixed processing and timing", tcap),
      table(["Item", "Value", "Where"],
            [["MI band-pass", "8\u201330 Hz, zero-phase", "eeg_training.MI_BAND"],
             ["CSP", "4 filters per CSP; one-vs-rest for 3 classes (12 features)", "eeg_models.MIModel"],
             ["Onset trim", "0.4 s at the start of each recorded MI block",
              "transient_ms in config/eeg.yaml (no GUI field)"],
             ["MI get-ready cue", "2 s, not recorded", "Record tab (MI)"],
             ["MI cue order", "Block-randomised: each class once per rep; no immediate repeats",
              "eeg_tab.mi_cue_order"],
             ["ErrP band-pass", "1\u201310 Hz, zero-phase", "eeg_training.ERRP_BAND"],
             ["ErrP epoch / baseline", "\u22120.2 to +0.8 s / \u22120.2 to 0 s", "ERRP_TMIN, ERRP_TMAX"],
             ["ErrP features", "12 time bins \u00d7 8 channels = 96", "eeg_models.ErrPModel"],
             ["ErrP classifier", "Shrinkage LDA, balanced priors", "eeg_models.ErrPModel"],
             ["ErrP task timing", "Fixation 0.7\u20131.1 s \u00b7 target 0.9 s \u00b7 outcome 1.0 s",
              "Record tab (ErrP)"],
             ["Cross-validation", "StratifiedGroupKFold, k = min(5, fewest trials in any class); "
              "final model refit on all data", "eeg_training._grouped_cv"],
             ["Live MI", "Raw window \u2192 8\u201330 Hz \u2192 model; update every \u2265 150 ms",
              "eeg_tab.EEGEngine"]],
            [100, 222, W_CONTENT - 322], bold_col=0)]

# References
refs = [
    "Pfurtscheller G, Lopes da Silva FH. Event-related EEG/MEG synchronization and desynchronization: "
    "basic principles. <i>Clin Neurophysiol</i>. 1999;110(11):1842\u20131857.",
    "Pfurtscheller G, Neuper C. Motor imagery and direct brain\u2013computer communication. "
    "<i>Proc IEEE</i>. 2001;89(7):1123\u20131134.",
    "Ramoser H, M\u00fcller-Gerking J, Pfurtscheller G. Optimal spatial filtering of single trial EEG "
    "during imagined hand movement. <i>IEEE Trans Rehabil Eng</i>. 2000;8(4):441\u2013446.",
    "Dornhege G, Blankertz B, Curio G, M\u00fcller K-R. Boosting bit rates in noninvasive EEG "
    "single-trial classifications by feature combination and multiclass paradigms. "
    "<i>IEEE Trans Biomed Eng</i>. 2004;51(6):993\u20131002.",
    "Lotte F, Bougrain L, Cichocki A, et al. A review of classification algorithms for EEG-based "
    "brain\u2013computer interfaces: a 10 year update. <i>J Neural Eng</i>. 2018;15(3):031005.",
    "Vidaurre C, Blankertz B. Towards a cure for BCI illiteracy. <i>Brain Topogr</i>. "
    "2010;23(2):194\u2013198.",
    "Chavarriaga R, Mill\u00e1n JdR. Learning from EEG error-related potentials in noninvasive "
    "brain\u2013computer interfaces. <i>IEEE Trans Neural Syst Rehabil Eng</i>. 2010;18(4):381\u2013388.",
    "Chavarriaga R, Sobolewski A, Mill\u00e1n JdR. Errare machinale est: the use of error-related "
    "potentials in brain\u2013machine interfaces. <i>Front Neurosci</i>. 2014;8:208.",
    "Blankertz B, Lemm S, Treder M, Haufe S, M\u00fcller K-R. Single-trial analysis and classification "
    "of ERP components \u2014 a tutorial. <i>NeuroImage</i>. 2011;56(2):814\u2013825.",
    "Varoquaux G, Raamana PR, Engemann DA, et al. Assessing and tuning brain decoders: "
    "cross-validation, caveats, and guidelines. <i>NeuroImage</i>. 2017;145:166\u2013179.",
]
S += [P("References", h1)] + [P(f"[{i}]&nbsp;&nbsp;{r}", ref_st) for i, r in enumerate(refs, 1)]

# ---------------- build ----------------
doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=20 * mm,
                      title="ERD/ERS, MI & ErrP \u2014 Training Methodology (rev. 2)",
                      author="biopipe", subject="EEG training methodology: 3-class MI + ErrP")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f",
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])
doc.build(S)
print("built", OUT)
