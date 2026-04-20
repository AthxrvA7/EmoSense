"""
report_generator.py

Generates a PDF report for a user's emotional analysis using ReportLab.

Sections:
  - Title & metadata (user ID, date/time)
  - Summary (average intensity, stability score, risk level, message)
  - Emotion statistics (counts)
  - Emotion distribution chart (bar)
  - Intensity trend chart (line)
  - Pattern insights (dominant emotion, top keywords)
"""

import os
import tempfile
from datetime import datetime

import matplotlib
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ================================
# COLOUR PALETTE
# ================================
_PRIMARY = HexColor("#2E86AB")
_DARK = HexColor("#1B1B2F")
_ACCENT = HexColor("#E27D60")
_LIGHT_BG = HexColor("#F5F5F5")
_WHITE = HexColor("#FFFFFF")
_GREY = HexColor("#666666")


# ================================
# CUSTOM STYLES
# ================================
def _build_styles():
    """Return a dictionary of ParagraphStyles used throughout the report."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=22,
            leading=28,
            textColor=_DARK,
            alignment=TA_CENTER,
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=_GREY,
            alignment=TA_CENTER,
            spaceAfter=6 * mm,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            textColor=_PRIMARY,
            spaceBefore=8 * mm,
            spaceAfter=3 * mm,
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=11,
            leading=16,
            textColor=_DARK,
        ),
        "label": ParagraphStyle(
            "Label",
            parent=base["Normal"],
            fontSize=11,
            leading=16,
            textColor=_GREY,
        ),
        "value": ParagraphStyle(
            "Value",
            parent=base["Normal"],
            fontSize=11,
            leading=16,
            textColor=_DARK,
        ),
    }
    return styles


# ================================
# HELPER BUILDERS
# ================================
def _divider():
    """Horizontal rule divider."""
    return HRFlowable(
        width="100%",
        thickness=0.5,
        color=HexColor("#CCCCCC"),
        spaceBefore=4 * mm,
        spaceAfter=4 * mm,
    )


def _summary_table(metrics, risk_level, message, styles):
    """Build a two-column summary table."""
    data = [
        [
            Paragraph("Average Intensity", styles["label"]),
            Paragraph(str(metrics.get("avg_intensity", "N/A")), styles["value"]),
        ],
        [
            Paragraph("Stability Score", styles["label"]),
            Paragraph(str(metrics.get("stability_score", "N/A")), styles["value"]),
        ],
        [
            Paragraph("Risk Level", styles["label"]),
            Paragraph(f"<b>{risk_level}</b>", styles["value"]),
        ],
        [
            Paragraph("Assessment", styles["label"]),
            Paragraph(message, styles["value"]),
        ],
    ]

    table = Table(data, colWidths=[55 * mm, 110 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _LIGHT_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, HexColor("#CCCCCC")),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]
        )
    )
    return table


def _emotion_table(emotion_counts, styles):
    """Build a table of emotion → count rows with a header."""
    header = [
        Paragraph("<b>Emotion</b>", styles["label"]),
        Paragraph("<b>Count</b>", styles["label"]),
    ]
    rows = [header]

    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        rows.append(
            [
                Paragraph(emotion.capitalize(), styles["value"]),
                Paragraph(str(count), styles["value"]),
            ]
        )

    table = Table(rows, colWidths=[80 * mm, 40 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
                ("BACKGROUND", (0, 1), (-1, -1), _LIGHT_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, HexColor("#DDDDDD")),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]
        )
    )
    return table


def _patterns_section(patterns, styles):
    """Return a list of flowables for the pattern-insights section."""
    elements = []

    dominant = patterns.get("dominant_emotion", "N/A")
    elements.append(
        Paragraph(
            f"<b>Dominant Emotion:</b>  {dominant.capitalize()}", styles["body"]
        )
    )
    elements.append(Spacer(1, 3 * mm))

    top_keywords = patterns.get("top_keywords", [])
    if top_keywords:
        kw_str = ", ".join(
            f"{word} ({count})" for word, count in top_keywords
        )
        elements.append(
            Paragraph(f"<b>Top Keywords:</b>  {kw_str}", styles["body"])
        )
    else:
        elements.append(
            Paragraph("<b>Top Keywords:</b>  None detected", styles["body"])
        )

    return elements


# ================================
# CHART BUILDERS
# ================================
_CHART_COLOURS = ["#2E86AB", "#E27D60", "#41B3A3", "#C38D9E", "#E8A87C",
                  "#85CDCA", "#D4A5A5", "#392F5A"]


def _plot_emotion_distribution(emotion_counts: dict, output_path: str) -> str:
    """
    Save a styled bar chart of emotion counts to *output_path* and return it.
    Uses FigureCanvasAgg directly — never touches the global pyplot backend.
    """
    emotions = [e.capitalize() for e in emotion_counts.keys()]
    counts = list(emotion_counts.values())
    colours = (_CHART_COLOURS * ((len(emotions) // len(_CHART_COLOURS)) + 1))[:len(emotions)]

    fig = Figure(figsize=(6, 3.2), dpi=150)
    FigureCanvasAgg(fig)          # attach Agg canvas without touching global state
    ax = fig.add_subplot(111)

    bars = ax.bar(emotions, counts, color=colours, edgecolor="white", linewidth=0.6)

    ax.set_title("Emotion Distribution", fontsize=13, fontweight="bold", pad=12,
                 color="#1B1B2F")
    ax.set_ylabel("Count", fontsize=10, color="#666666")
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=9, colors="#333333")
    ax.tick_params(axis="y", labelsize=9, colors="#666666")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=8, color="#333333")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=False)
    return output_path


def _plot_intensity_trend(intensity_timeline: list, output_path: str) -> str:
    """
    Save a styled line chart of intensity over message index.
    Uses FigureCanvasAgg directly — never touches the global pyplot backend.
    """
    indices = list(range(1, len(intensity_timeline) + 1))

    fig = Figure(figsize=(6, 3.2), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.plot(indices, intensity_timeline, color="#2E86AB", linewidth=2,
            marker="o", markersize=4, markerfacecolor="#E27D60",
            markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(indices, intensity_timeline, alpha=0.10, color="#2E86AB")

    ax.set_title("Emotion Intensity Over Time", fontsize=13, fontweight="bold",
                 pad=12, color="#1B1B2F")
    ax.set_xlabel("Message #", fontsize=10, color="#666666")
    ax.set_ylabel("Intensity", fontsize=10, color="#666666")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="both", labelsize=9, colors="#666666")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=False)
    return output_path


# ================================
# PUBLIC API
# ================================
def generate_report(user_id: str, analysis_result: dict, output_dir: str = ".") -> str:
    """
    Generate a PDF report and return the file path.

    Parameters
    ----------
    user_id : str
        Identifier for the user.
    analysis_result : dict
        Pre-computed analysis data with the structure:
        {
            "metrics": {
                "avg_intensity": float,
                "emotion_counts": dict,
                "stability_score": float,
            },
            "patterns": {
                "dominant_emotion": str,
                "top_keywords": list[tuple[str, int]],
                ...
            },
            "risk_level": str,
            "message": str,
            "intensity_timeline": list[float],  # optional
        }
    output_dir : str
        Directory where the PDF will be saved.  Defaults to the
        current working directory.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """

    # --- Unpack analysis data ---
    metrics = analysis_result.get("metrics", {})
    patterns = analysis_result.get("patterns", {})
    risk_level = analysis_result.get("risk_level", "N/A")
    message = analysis_result.get("message", "")
    emotion_counts = metrics.get("emotion_counts", {})
    intensity_timeline = analysis_result.get("intensity_timeline", [])

    # Temp files for chart images (cleaned up after build)
    _temp_files = []

    # --- File path ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"EmoSense_Report_{user_id}_{timestamp_str}.pdf"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # --- Document setup ---
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
    )

    styles = _build_styles()
    story = []

    # ---- Title ----
    story.append(Paragraph("EmoSense Emotional Analysis Report", styles["title"]))

    now = datetime.now().strftime("%B %d, %Y  •  %I:%M %p")
    story.append(
        Paragraph(f"User ID: <b>{user_id}</b>  |  Generated: {now}", styles["subtitle"])
    )
    story.append(_divider())

    # ---- Summary ----
    story.append(Paragraph("Summary", styles["heading"]))
    story.append(_summary_table(metrics, risk_level, message, styles))

    # ---- Emotion Statistics ----
    story.append(Paragraph("Emotion Statistics", styles["heading"]))
    if emotion_counts:
        story.append(_emotion_table(emotion_counts, styles))
    else:
        story.append(Paragraph("No emotion data available.", styles["body"]))

    # ---- Emotion Distribution Chart ----
    if emotion_counts:
        story.append(Paragraph("Emotion Distribution", styles["heading"]))
        bar_path = os.path.join(
            tempfile.gettempdir(),
            f"emosense_bar_{user_id}_{datetime.now().strftime('%H%M%S')}.png",
        )
        _plot_emotion_distribution(emotion_counts, bar_path)
        _temp_files.append(bar_path)
        story.append(Image(bar_path, width=160 * mm, height=86 * mm))

    # ---- Intensity Trend Chart ----
    if intensity_timeline and len(intensity_timeline) >= 2:
        story.append(Paragraph("Intensity Trend", styles["heading"]))
        line_path = os.path.join(
            tempfile.gettempdir(),
            f"emosense_line_{user_id}_{datetime.now().strftime('%H%M%S')}.png",
        )
        _plot_intensity_trend(intensity_timeline, line_path)
        _temp_files.append(line_path)
        story.append(Image(line_path, width=160 * mm, height=86 * mm))

    # ---- Pattern Insights ----
    story.append(Paragraph("Pattern Insights", styles["heading"]))
    story.extend(_patterns_section(patterns, styles))

    # ---- Footer divider ----
    story.append(Spacer(1, 12 * mm))
    story.append(_divider())
    story.append(
        Paragraph(
            "This report was generated automatically by EmoSense. "
            "It is intended for informational purposes only and should not be "
            "used as a substitute for professional mental health advice.",
            ParagraphStyle(
                "Disclaimer",
                fontSize=8,
                leading=11,
                textColor=_GREY,
                alignment=TA_CENTER,
            ),
        )
    )

    # ---- Build PDF ----
    doc.build(story)

    # ---- Cleanup temp chart images ----
    for tmp in _temp_files:
        try:
            os.remove(tmp)
        except OSError:
            pass

    return os.path.abspath(filepath)


# ================================
# TEST BLOCK
# ================================
if __name__ == "__main__":
    sample_result = {
        "metrics": {
            "avg_intensity": 0.68,
            "emotion_counts": {
                "happy": 12,
                "sad": 8,
                "anxious": 5,
                "angry": 3,
                "neutral": 7,
            },
            "stability_score": 0.74,
        },
        "patterns": {
            "dominant_emotion": "happy",
            "repeated_negative": 16,
            "top_keywords": [
                ("stress", 9),
                ("work", 7),
                ("tired", 5),
                ("family", 4),
                ("sleep", 3),
            ],
            "night_stress": 4,
        },
        "risk_level": "Moderate",
        "message": "User shows moderate emotional fluctuations.",
        "intensity_timeline": [
            0.45, 0.52, 0.70, 0.65, 0.80, 0.73, 0.60,
            0.55, 0.78, 0.82, 0.68, 0.90, 0.74, 0.61, 0.50,
        ],
    }

    path = generate_report("test_user_01", sample_result)
    print(f"Report saved -> {path}")
