"""Reproducible generator for the PsychScanner logo/brand system.

Regenerates every logo (framework repos, library repos, docker image
badges) as SVG + PDF from one shared set of color tokens, so all marks
stay visually consistent. Run: `python3 branding_assets.py`.

Color story (extends ps_ps-primal_logo.jpeg's existing maroon/blue split):
  psychscanner            -> MAROON       (core framework)
  psychscanner-primal     -> BLUE         (Prime Intellect Hub variant)
  psyscan-library         -> MAROON_TINT, dashed border (data index, not code)
  psyscan-library-primal  -> BLUE_TINT,   dashed border
  docker: ps-general      -> DOCKER_CORE   (standard build)
  docker: ps-neuroscanner -> DOCKER_INTERP (nnsight/nnterp/vlm extras)
  docker: ps-primal       -> BLUE          (matches psychscanner-primal)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "logos")
os.makedirs(OUT, exist_ok=True)

MAROON = "#7A1B4A"
MAROON_TINT = "#B05E8C"
BLUE = "#1E88C4"
BLUE_TINT = "#6FBBE0"
DOCKER_CORE = "#1F4E79"
DOCKER_INTERP = "#6B21A8"
INK = "#2B2B2B"

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"


def _save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.svg"), transparent=True, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _bracket(ax, x, color, flip=False, height=1.0, y0=0.0):
    w = 0.12 * (-1 if flip else 1)
    lw = 3.2
    ax.plot([x, x + w, x + w], [y0 + height, y0 + height, y0], color=color, lw=lw, solid_capstyle="butt")
    ax.plot([x, x + w, x + w], [y0, y0, y0], color=color, lw=lw, solid_capstyle="butt")
    ax.plot([x, x], [y0, y0 + height], color=color, lw=lw, solid_capstyle="butt")


def _bracket_ls(ax, x, color, flip=False, height=1.0, y0=0.0, lw=3.2, ls="solid"):
    w = 0.09 * (-1 if flip else 1)
    ax.plot([x, x + w, x + w], [y0 + height, y0 + height, y0], color=color, lw=lw,
            linestyle=ls, solid_capstyle="butt")
    ax.plot([x, x + w], [y0, y0], color=color, lw=lw, linestyle=ls, solid_capstyle="butt")
    ax.plot([x, x], [y0, y0 + height], color=color, lw=lw, linestyle=ls, solid_capstyle="butt")


def _brain_circuit_icon(ax, cx, cy, r, brain_color, circuit_color):
    """Half brain-outline / half circuit-node glyph, matching the source jpeg."""
    import math
    outline = mpatches.Arc((cx, cy), 2 * r, 2 * r, theta1=90, theta2=270,
                            edgecolor=brain_color, lw=2.2)
    ax.add_patch(outline)
    ax.plot([cx, cx], [cy - r, cy + r], color=brain_color, lw=2.2)
    # gyri wrinkles: short scalloped arcs suggesting brain folds
    for k, (frac, span) in enumerate([(0.55, 70), (0.15, 100), (-0.35, 80), (-0.7, 50)]):
        cy_k = cy + frac * r
        wrinkle = mpatches.Arc((cx - r * 0.15, cy_k), r * 0.85, r * 0.4,
                                theta1=100, theta2=260, edgecolor=brain_color, lw=1.1)
        ax.add_patch(wrinkle)
    # circuit lattice (right)
    import math
    nodes = [(cx + r * 0.15, cy + r * 0.55), (cx + r * 0.55, cy + r * 0.75), (cx + r * 0.85, cy + r * 0.35),
              (cx + r * 0.55, cy - r * 0.05), (cx + r * 0.85, cy - r * 0.45), (cx + r * 0.35, cy - r * 0.65)]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (3, 5)]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=circuit_color, lw=1.4)
    for (nx, ny) in nodes:
        ax.add_patch(mpatches.Circle((nx, ny), r * 0.06, facecolor=circuit_color, edgecolor="none"))


def make_wordmark(name, label, color, tint=None, tagline=None, with_icon=False):
    fig, ax = plt.subplots(figsize=(6, 1.6 if not with_icon else 2.2))
    ax.axis("off")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.6 if not with_icon else 2.2)
    y0 = 0.2 if not with_icon else 0.5
    x_text = 0.35
    if with_icon:
        _brain_circuit_icon(ax, 0.55, y0 + 0.6, 0.5, color, tint or BLUE)
        x_text = 1.25
    _bracket(ax, x_text - 0.15, color, flip=False, height=1.0, y0=y0)
    ax.text(x_text, y0 + 0.5, label, fontsize=30, fontweight="bold", fontstyle="italic",
            color=color, ha="left", va="center")
    text_w = 0.34 * len(label)
    _bracket(ax, x_text + text_w + 0.15, color, flip=True, height=1.0, y0=y0)
    if tagline:
        ax.text(x_text, y0 - 0.18, tagline.upper(), fontsize=8, color="#555555",
                ha="left", va="center", family="monospace")
    _save(fig, name)


def _measured_text_width(fig, ax, text_obj):
    """Actual rendered text width in data coords (guessing a char-width
    constant broke on every label length -- measure instead)."""
    fig.canvas.draw()
    bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
    (x0, _), (x1, _) = ax.transData.inverted().transform(bbox)
    return x1 - x0


def make_docker_badge(name, tag_label, color, dashed=False):
    """Small bracket wordmark, same family as make_wordmark but scaled down --
    these are meant to sit as compact badges, not full logos."""
    fig, ax = plt.subplots(figsize=(2.6, 0.75))
    ax.axis("off")
    y0 = 0.14
    x_text = 0.24
    txt = ax.text(x_text, y0 + 0.24, tag_label, fontsize=14, fontweight="bold", fontstyle="italic",
                   color=color, ha="left", va="center")
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 0.75)
    text_w = _measured_text_width(fig, ax, txt)
    ls = (0, (2, 1.5)) if dashed else "solid"
    _bracket_ls(ax, x_text - 0.1, color, flip=False, height=0.48, y0=y0, lw=2.2, ls=ls)
    _bracket_ls(ax, x_text + text_w + 0.1, color, flip=True, height=0.48, y0=y0, lw=2.2, ls=ls)
    fig.set_size_inches(x_text + text_w + 0.35, 0.75)
    ax.set_xlim(0, x_text + text_w + 0.35)
    _save(fig, name)


def make_library_mark(name, label, color, tint):
    """Same bracket-wordmark family as the framework repos (make_wordmark),
    dashed brackets to signal "index/data, not code"."""
    fig, ax = plt.subplots(figsize=(4.6, 1.1))
    ax.axis("off")
    y0 = 0.18
    x_text = 0.3
    txt = ax.text(x_text, y0 + 0.35, label, fontsize=19, fontweight="bold", fontstyle="italic",
                   color=color, ha="left", va="center")
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 1.1)
    text_w = _measured_text_width(fig, ax, txt)
    _bracket_ls(ax, x_text - 0.12, color, flip=False, height=0.7, y0=y0, lw=2.4, ls=(0, (3, 2)))
    _bracket_ls(ax, x_text + text_w + 0.12, color, flip=True, height=0.7, y0=y0, lw=2.4, ls=(0, (3, 2)))
    fig.set_size_inches(x_text + text_w + 0.45, 1.1)
    ax.set_xlim(0, x_text + text_w + 0.45)
    _save(fig, name)


if __name__ == "__main__":
    make_wordmark("psychscanner", "psychscanner", MAROON)
    make_wordmark("psychscanner-primal", "psychscanner-primal", MAROON,
                  tint=BLUE, tagline="optimized for prime intellect environment hub", with_icon=True)
    make_docker_badge("docker_ps-general", "ps-general", DOCKER_CORE)
    make_docker_badge("docker_ps-neuroscanner", "ps-neuroscanner", DOCKER_INTERP)
    make_docker_badge("docker_ps-primal", "ps-primal", BLUE)
    make_library_mark("psyscan-library", "psyscan-library", MAROON_TINT, MAROON)
    make_library_mark("psyscan-library-primal", "psyscan-library-primal", BLUE_TINT, BLUE)
    print("wrote logos to", os.path.abspath(OUT))
