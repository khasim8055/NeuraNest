# app/ui/analytics_panel.py
# ================================================================
# NeuraCare — Analytics Dashboard Panel
# ================================================================
# 5 charts embedded in PyQt6 using matplotlib FigureCanvasQTAgg.
# Accessible from a dedicated Analytics tab in the main window.
#
# Layout:
#   Header row   — 4 KPI stat cards
#   Row 1        — Diagnosis frequency (wide) | Age distribution
#   Row 2        — LOS by diagnosis | Monthly admissions | Risk pie
# ================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.analytics import (
    get_diagnosis_data, get_age_data, get_los_data,
    get_monthly_data, get_risk_data, get_summary_stats,
)

# ── Colour palette ────────────────────────────────────────────────
COLORS = {
    "bg_dark":      "#1A1D2E",
    "bg_panel":     "#22253A",
    "bg_card":      "#2A2D42",
    "accent_blue":  "#4A90D9",
    "accent_green": "#52B788",
    "accent_amber": "#F4A50A",
    "accent_red":   "#D64545",
    "text_primary": "#E8EAF6",
    "text_muted":   "#8B90A8",
    "border":       "#35384F",
}

# ── Chart colours ─────────────────────────────────────────────────
CHART_BG   = "#22253A"
CHART_FG   = "#E8EAF6"
CHART_MUTED = "#8B90A8"
CHART_BLUE  = "#4A90D9"
CHART_GREEN = "#52B788"
CHART_AMBER = "#F4A50A"
CHART_RED   = "#D64545"
CHART_TEAL  = "#4CC9C9"


# ================================================================
# MATPLOTLIB CANVAS WRAPPER
# ================================================================

class ChartCanvas(FigureCanvasQTAgg):
    """Thin wrapper around FigureCanvasQTAgg for embedding charts."""

    def __init__(self, fig: Figure):
        super().__init__(fig)
        self.setStyleSheet(f"background-color: {COLORS['bg_card']};")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )


def _chart_fig(w: float = 5, h: float = 3) -> Figure:
    """Create a matplotlib Figure with our dark background."""
    fig = Figure(figsize=(w, h), facecolor=CHART_BG)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)
    return fig


def _style_ax(ax, title: str = ""):
    """Apply consistent dark theme to an axes."""
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_MUTED, labelsize=8)
    ax.xaxis.label.set_color(CHART_MUTED)
    ax.yaxis.label.set_color(CHART_MUTED)
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
    ax.grid(True, color=COLORS["border"], linewidth=0.4, alpha=0.6)
    if title:
        ax.set_title(title, color=CHART_FG, fontsize=10,
                     fontweight="bold", pad=8)


def _no_data_fig(message: str = "Add patients to see analytics") -> Figure:
    """Return a figure with a 'no data' message."""
    fig = _chart_fig()
    ax  = fig.add_subplot(111)
    ax.set_facecolor(CHART_BG)
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", color=CHART_MUTED,
            fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ================================================================
# INDIVIDUAL CHART BUILDERS
# ================================================================

def build_diagnosis_chart() -> Figure:
    data = get_diagnosis_data(top_n=8)
    if not data["enough_data"]:
        return _no_data_fig("No diagnosis data yet")

    labels = [l[:20] + "…" if len(l) > 20 else l for l in data["labels"]]
    counts = data["counts"]

    fig = _chart_fig(w=6, h=3)
    ax  = fig.add_subplot(111)
    bars = ax.bar(range(len(labels)), counts, color=CHART_BLUE,
                  width=0.6, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8,
                       color=CHART_MUTED)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                str(int(h)), ha="center", va="bottom",
                fontsize=8, color=CHART_FG, fontweight="bold")

    _style_ax(ax, "Diagnosis Frequency")
    ax.set_ylabel("Patients", fontsize=8)
    ax.tick_params(axis="x", length=0)
    return fig


def build_age_chart() -> Figure:
    data = get_age_data()
    if not data["enough_data"]:
        return _no_data_fig("No age data yet")

    fig = _chart_fig(w=4, h=3)
    ax  = fig.add_subplot(111)
    ax.bar(range(len(data["buckets"])), data["counts"],
           color=CHART_TEAL, width=0.65, zorder=3)
    ax.set_xticks(range(len(data["buckets"])))
    ax.set_xticklabels(data["buckets"], fontsize=8, color=CHART_MUTED)
    _style_ax(ax, f"Age Distribution (avg {data['avg_age']} yrs)")
    ax.set_ylabel("Patients", fontsize=8)
    return fig


def build_los_chart() -> Figure:
    data = get_los_data(top_n=6)
    if not data["enough_data"]:
        return _no_data_fig("No LOS data yet")

    diags   = [d[:18] + "…" if len(d) > 18 else d for d in data["diagnoses"]]
    avg_los = data["avg_los"]
    overall = data["overall_avg"]

    colors = [CHART_GREEN if v <= overall else CHART_AMBER for v in avg_los]

    fig = _chart_fig(w=4, h=3)
    ax  = fig.add_subplot(111)
    ax.barh(range(len(diags)), avg_los, color=colors, height=0.55, zorder=3)
    ax.set_yticks(range(len(diags)))
    ax.set_yticklabels(diags, fontsize=8, color=CHART_MUTED)
    ax.axvline(overall, color=CHART_RED, linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=4)
    ax.text(overall + 0.1, len(diags) - 0.6,
            f"avg {overall}d", fontsize=7, color=CHART_RED)
    _style_ax(ax, "Avg LOS by Diagnosis (days)")
    ax.set_xlabel("Days", fontsize=8)
    return fig


def build_monthly_chart() -> Figure:
    data = get_monthly_data()
    if not data["enough_data"]:
        return _no_data_fig("Need 2+ months of data")

    months  = data["months"]
    counts  = data["counts"]
    avg_los = data["avg_los"]

    fig = _chart_fig(w=5, h=3)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    x = range(len(months))
    ax1.bar(x, counts, color=CHART_BLUE, width=0.5,
            alpha=0.8, zorder=3, label="Admissions")
    ax2.plot(x, avg_los, color=CHART_AMBER, linewidth=2,
             marker="o", markersize=5, zorder=4, label="Avg LOS")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(months, rotation=30, ha="right",
                        fontsize=7, color=CHART_MUTED)
    ax1.set_ylabel("Admissions", fontsize=8, color=CHART_BLUE)
    ax2.set_ylabel("Avg LOS (days)", fontsize=8, color=CHART_AMBER)
    ax2.tick_params(colors=CHART_MUTED, labelsize=8)
    ax2.spines["right"].set_color(COLORS["border"])
    _style_ax(ax1, "Monthly Admissions & Avg LOS")
    return fig


def build_risk_chart() -> Figure:
    data = get_risk_data()
    if not data["enough_data"]:
        return _no_data_fig("No risk data yet")

    labels = data["labels"]
    counts = data["counts"]
    colors_map = {
        "High":   CHART_RED,
        "Medium": CHART_AMBER,
        "Low":    CHART_GREEN,
    }
    pie_colors = [colors_map[l] for l in labels]
    # Only include non-zero slices
    nonzero = [(l, c, col) for l, c, col in zip(labels, counts, pie_colors) if c > 0]
    if not nonzero:
        return _no_data_fig("No risk data yet")

    nz_labels, nz_counts, nz_colors = zip(*nonzero)

    fig = _chart_fig(w=4, h=3)
    ax  = fig.add_subplot(111)
    wedges, texts, autotexts = ax.pie(
        nz_counts,
        colors=nz_colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"edgecolor": CHART_BG, "linewidth": 2},
    )
    for t in texts + autotexts:
        t.set_color(CHART_FG)
        t.set_fontsize(9)

    legend_patches = [
        mpatches.Patch(color=colors_map[l], label=f"{l}: {c}")
        for l, c in zip(nz_labels, nz_counts)
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=8, framealpha=0,
              labelcolor=CHART_FG)

    ax.set_facecolor(CHART_BG)
    ax.set_title(f"Risk Breakdown (n={data['total']})",
                 color=CHART_FG, fontsize=10, fontweight="bold", pad=8)
    return fig


# ================================================================
# STAT CARD WIDGET
# ================================================================

def make_stat_card(title: str, value: str, subtitle: str = "",
                   color: str = None) -> QFrame:
    """Create a KPI stat card widget."""
    card = QFrame()
    card.setStyleSheet(f"""
        QFrame {{
            background-color: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
        }}
    """)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 12, 16, 12)
    layout.setSpacing(4)

    title_label = QLabel(title)
    title_label.setStyleSheet(
        f"color: {COLORS['text_muted']}; font-size: 10px; "
        f"font-weight: 500; border: none;"
    )

    val_color = color or COLORS["accent_blue"]
    value_label = QLabel(value)
    value_label.setStyleSheet(
        f"color: {val_color}; font-size: 22px; "
        f"font-weight: bold; border: none;"
    )

    layout.addWidget(title_label)
    layout.addWidget(value_label)

    if subtitle:
        sub_label = QLabel(subtitle)
        sub_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; border: none;"
        )
        layout.addWidget(sub_label)

    return card


# ================================================================
# ANALYTICS PANEL WIDGET
# ================================================================

class AnalyticsPanel(QWidget):
    """
    Full analytics dashboard.
    Can be shown as a full-screen overlay or in the center panel.
    """

    def __init__(self, on_close=None):
        super().__init__()
        self.on_close = on_close
        self._canvases = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ───────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        header.setFixedHeight(56)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Practice Analytics")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_blue']}; }}
        """)
        refresh_btn.setFixedHeight(32)
        refresh_btn.clicked.connect(self.refresh)

        if self.on_close:
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet(refresh_btn.styleSheet())
            close_btn.setFixedHeight(32)
            close_btn.clicked.connect(self.on_close)
            header_layout.addWidget(close_btn)
            header_layout.addSpacing(8)

        header_layout.addWidget(title, stretch=1)
        header_layout.addWidget(refresh_btn)

        # ── Scrollable body ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; "
            f"background-color: {COLORS['bg_dark']}; }}"
        )

        self.body = QWidget()
        self.body.setStyleSheet(
            f"background-color: {COLORS['bg_dark']};"
        )
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(20, 16, 20, 20)
        self.body_layout.setSpacing(16)

        scroll.setWidget(self.body)
        outer.addWidget(header)
        outer.addWidget(scroll, stretch=1)

        # Build initial charts
        self._build_charts()

    def _clear_body(self):
        """Remove all widgets from body layout."""
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Close all matplotlib figures to free memory
        for canvas in self._canvases:
            plt.close(canvas.figure)
        self._canvases.clear()

    def _build_charts(self):
        """Build all stat cards and charts."""
        # ── Stat cards row ────────────────────────────────────────
        stats = get_summary_stats()
        stat_row = QHBoxLayout()
        stat_row.setSpacing(12)

        stat_row.addWidget(make_stat_card(
            "Total Patients",
            str(stats["total_patients"]),
            color=COLORS["accent_blue"],
        ))
        stat_row.addWidget(make_stat_card(
            "Avg Length of Stay",
            f"{stats['avg_los']} days",
            color=COLORS["accent_green"],
        ))
        stat_row.addWidget(make_stat_card(
            "High Risk Patients",
            str(stats["high_risk_count"]),
            subtitle=f"{stats['high_risk_pct']}% of total",
            color=COLORS["accent_red"],
        ))
        stat_row.addWidget(make_stat_card(
            "Medium + Low Risk",
            str(stats["total_patients"] - stats["high_risk_count"]),
            subtitle="Stable for discharge",
            color=COLORS["accent_amber"],
        ))

        stat_widget = QWidget()
        stat_widget.setStyleSheet("background: transparent;")
        stat_widget.setLayout(stat_row)
        self.body_layout.addWidget(stat_widget)

        # ── Chart row 1: Diagnosis + Age ─────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        diag_canvas = ChartCanvas(build_diagnosis_chart())
        diag_canvas.setMinimumHeight(220)
        self._canvases.append(diag_canvas)

        age_canvas = ChartCanvas(build_age_chart())
        age_canvas.setMinimumHeight(220)
        self._canvases.append(age_canvas)

        row1.addWidget(self._wrap_chart(diag_canvas, ""), stretch=3)
        row1.addWidget(self._wrap_chart(age_canvas, ""), stretch=2)

        row1_widget = QWidget()
        row1_widget.setStyleSheet("background: transparent;")
        row1_widget.setLayout(row1)
        self.body_layout.addWidget(row1_widget)

        # ── Chart row 2: LOS + Monthly + Risk ────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(12)

        los_canvas = ChartCanvas(build_los_chart())
        los_canvas.setMinimumHeight(220)
        self._canvases.append(los_canvas)

        monthly_canvas = ChartCanvas(build_monthly_chart())
        monthly_canvas.setMinimumHeight(220)
        self._canvases.append(monthly_canvas)

        risk_canvas = ChartCanvas(build_risk_chart())
        risk_canvas.setMinimumHeight(220)
        self._canvases.append(risk_canvas)

        row2.addWidget(self._wrap_chart(los_canvas, ""), stretch=2)
        row2.addWidget(self._wrap_chart(monthly_canvas, ""), stretch=2)
        row2.addWidget(self._wrap_chart(risk_canvas, ""), stretch=2)

        row2_widget = QWidget()
        row2_widget.setStyleSheet("background: transparent;")
        row2_widget.setLayout(row2)
        self.body_layout.addWidget(row2_widget)

    def _wrap_chart(self, canvas: ChartCanvas, title: str) -> QFrame:
        """Wrap a chart canvas in a styled card frame."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(canvas)
        return card

    def refresh(self):
        """Reload all data and redraw charts."""
        self._clear_body()
        self._build_charts()
