#!/usr/bin/env python3
"""Generates docs/level-set-representations.svg and docs/level-set-update.svg: one level set vs. MOGAC.

(a) the single-object level set: a signed distance phi at every grid node (negative inside), its
    narrow band and zero layer; (b) MOGAC: the Enright disk split into four regions, every node
    storing a region label and the unsigned distance d to the nearest region boundary, with the
    band around every boundary (including the ones between regions); (c), (d) the same grid row as
    a cross-section: phi(x) against d(x), its labels, and a region's signed level set rebuilt from
    them. level-set-update.svg contrasts one update of a node on each. Colors match the enright2d panels. Run from anywhere:
    python3 docs/make_level_set_representation_diagram.py
"""
import math
from pathlib import Path

# Palette (same as apps/enright2d/Renderer.cpp and the other diagrams).
BACKGROUND = "#202024"
PANEL = "#141418"
GRID = "#464652"
TEXT = "#e6e6e6"
DIM_TEXT = "#a8a8b0"
CONTOUR = "#008080"
CONTOUR_FILL = "rgba(0,128,128,0.45)"
VERTEX = "#ffcc00"
BAND = "rgba(255,204,0,0.10)"
REGION = ["#e65a5a", "#5ab4e6", "#78d26e", "#ebbe46"]  # labels 1..4 (kRegionColors)
BACKGROUND_LABEL = "#6e6e78"
FONT = "Helvetica, Arial, sans-serif"

N = 16  # cells across
CELL = 28
GRID_SIZE = N * CELL
CENTER = (8.0, 8.0)
RADIUS = 5.3
ROW = 6  # the cross-section row (grid units, y down)
BAND_WIDTH = 3.5  # |phi| <= 3.5 is the band (NarrowBand2D::kMaxDistance)

WIDTH, HEIGHT = 1100, 930
MARGIN = 40
PANEL_A = (MARGIN + 30, 90)
PANEL_B = (WIDTH // 2 + 30, 90)
PLOT_TOP = 90 + GRID_SIZE + 90
PLOT_H = 200

out = []


def emit(s):
    out.append(s)


def paint(attr, color):
    """attr="color" for hex colors; attr + attr-opacity for rgba(...), which not every SVG renderer accepts."""
    if color.startswith("rgba("):
        r, g, b, a = (float(v) for v in color[5:-1].split(","))
        return f'{attr}="#{int(r):02x}{int(g):02x}{int(b):02x}" {attr}-opacity="{a}"'
    return f'{attr}="{color}"'


def text(x, y, s, size=14, color=TEXT, anchor="start", weight="normal", italic=False):
    style = ' font-style="italic"' if italic else ""
    emit(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" {paint("fill", color)} text-anchor="{anchor}" '
         f'font-weight="{weight}"{style}>{s}</text>')


def line(a, b, color, width, dash=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    emit(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" {paint("stroke", color)} '
         f'stroke-width="{width}" stroke-linecap="{cap}"{d}/>')


def circle(c, r, fill, stroke=None, width=0):
    s = f' {paint("stroke", stroke)} stroke-width="{width}"' if stroke else ""
    emit(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r}" {paint("fill", fill)}{s}/>')


def rect(x, y, w, h, fill, stroke=None, width=0, rx=0):
    s = f' {paint("stroke", stroke)} stroke-width="{width}"' if stroke else ""
    emit(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" {paint("fill", fill)}{s}/>')


def polyline(points, color, width, dash=None, fill="none"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    f = paint("fill", fill) if fill != "none" else 'fill="none"'
    emit(f'<polyline points="{pts}" {f} {paint("stroke", color)} stroke-width="{width}" stroke-linejoin="round"{d}/>')


# --- The two representations (grid units, y down; labels counter-clockwise from the upper right). ---

def phi(x, y):
    """Signed distance to the circle, negative inside."""
    return math.hypot(x - CENTER[0], y - CENTER[1]) - RADIUS


def segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    t = max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
    return math.hypot(p[0] - (ax + t * (bx - ax)), p[1] - (ay + t * (by - ay)))


def label(x, y):
    """Region label: 1..4 quadrants of the disk (1 upper right, counter-clockwise), 0 outside."""
    if phi(x, y) >= 0.0:
        return 0
    right, up = x >= CENTER[0], y < CENTER[1]
    return {(True, True): 1, (False, True): 2, (False, False): 3, (True, False): 4}[(right, up)]


def distance(x, y):
    """Unsigned distance to the nearest region boundary: the circle and both diameters."""
    cx, cy = CENTER
    return min(abs(phi(x, y)),
               segment_distance((x, y), (cx - RADIUS, cy), (cx + RADIUS, cy)),
               segment_distance((x, y), (cx, cy - RADIUS), (cx, cy + RADIUS)))


def region_color(l):
    return REGION[l - 1] if l > 0 else BACKGROUND_LABEL


def rgba(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def to_screen(origin, x, y):
    return origin[0] + x * CELL, origin[1] + y * CELL


def panel_frame(origin, title, subtitle):
    rect(origin[0] - 14, origin[1] - 58, GRID_SIZE + 28, GRID_SIZE + 72, PANEL, rx=6)
    text(origin[0], origin[1] - 34, title, 17, TEXT, weight="bold")
    text(origin[0], origin[1] - 14, subtitle, 12.5, DIM_TEXT)


def grid_lines(origin):
    for i in range(N + 1):
        line(to_screen(origin, i, 0), to_screen(origin, i, N), GRID, 0.8, cap="butt")
        line(to_screen(origin, 0, i), to_screen(origin, N, i), GRID, 0.8, cap="butt")


def band_cells(origin, value):
    """Shades every cell with a corner in the band."""
    for j in range(N):
        for i in range(N):
            if min(abs(value(i + a, j + b)) for a in (0, 1) for b in (0, 1)) <= BAND_WIDTH:
                x, y = to_screen(origin, i, j)
                rect(x, y, CELL, CELL, BAND)


def node_marks(origin, value, color_of):
    """Every node: a dot in its color, ringed when it is on the zero layer (|value| <= 1/2)."""
    for j in range(N + 1):
        for i in range(N + 1):
            v = value(i, j)
            p = to_screen(origin, i, j)
            in_band = abs(v) <= BAND_WIDTH
            circle(p, 3.4 if in_band else 2.0, color_of(i, j) if in_band else GRID)
            if abs(v) <= 0.5:
                circle(p, 6.0, "none", VERTEX, 1.6)


def section_row(origin):
    """The cross-section row plotted in (c) and (d)."""
    line(to_screen(origin, 0, ROW), to_screen(origin, N, ROW), DIM_TEXT, 1.2, dash="5,4", cap="butt")


emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
     f'font-family="{FONT}">')
rect(0, 0, WIDTH, HEIGHT, BACKGROUND)

# (a) One level set.
A = PANEL_A
panel_frame(A, "(a) One level set", "a signed distance φ at every node (negative inside)")
band_cells(A, phi)
grid_lines(A)
cx, cy = to_screen(A, *CENTER)
circle((cx, cy), RADIUS * CELL, CONTOUR_FILL, CONTOUR, 3)
node_marks(A, phi, lambda i, j: CONTOUR if phi(i, j) < 0 else DIM_TEXT)
section_row(A)
text(cx, cy + 4, "φ &lt; 0", 15, TEXT, anchor="middle", weight="bold")
text(A[0] + 8, A[1] + GRID_SIZE - 10, "φ &gt; 0", 15, TEXT, weight="bold")

# (b) MOGAC.
B = PANEL_B
panel_frame(B, "(b) MOGAC: many objects in one level set", "a region label (node color) and the unsigned distance d at every node")
band_cells(B, distance)
grid_lines(B)
cx, cy = to_screen(B, *CENTER)
r = RADIUS * CELL
# Quarter disks, counter-clockwise from the upper right (SVG y is down).
quarters = {1: (0, -90), 2: (-90, -180), 3: (180, 90), 4: (90, 0)}
for l, (a0, a1) in quarters.items():
    p0 = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
    p1 = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
    emit(f'<path d="M {cx:.1f} {cy:.1f} L {p0[0]:.1f} {p0[1]:.1f} A {r:.1f} {r:.1f} 0 0 0 {p1[0]:.1f} {p1[1]:.1f} Z" '
         f'{paint("fill", rgba(REGION[l - 1], 0.30))}/>')
circle((cx, cy), r, "none", TEXT, 2.4)
line((cx - r, cy), (cx + r, cy), TEXT, 2.4)
line((cx, cy - r), (cx, cy + r), TEXT, 2.4)
node_marks(B, distance, lambda i, j: region_color(label(i, j)))
section_row(B)
for l, (dx, dy) in {1: (1, -1), 2: (-1, -1), 3: (-1, 1), 4: (1, 1)}.items():
    text(cx + dx * 0.36 * r, cy + dy * 0.36 * r + 15 + (6 if dy < 0 else 0), f"{l}", 22, TEXT, anchor="middle", weight="bold")
text(B[0] + 8, B[1] + GRID_SIZE - 10, "label 0 (background)", 13, TEXT, weight="bold")


# (c), (d) Cross-sections along the dashed row.
def plot_frame(x0, title, subtitle, ymin, ymax):
    w = GRID_SIZE
    rect(x0 - 14, PLOT_TOP - 58, w + 28, PLOT_H + 112, PANEL, rx=6)
    text(x0, PLOT_TOP - 34, title, 17, TEXT, weight="bold")
    text(x0, PLOT_TOP - 14, subtitle, 12.5, DIM_TEXT)

    def to_plot(x, v):
        return x0 + x * CELL, PLOT_TOP + (ymax - v) / (ymax - ymin) * PLOT_H

    # Band, axis, ticks.
    y_hi, y_lo = to_plot(0, min(BAND_WIDTH, ymax))[1], to_plot(0, max(-BAND_WIDTH, ymin))[1]
    rect(x0, y_hi, w, y_lo - y_hi, BAND)
    for v in range(math.ceil(ymin), math.floor(ymax) + 1):
        y = to_plot(0, v)[1]
        line((x0, y), (x0 + w, y), GRID, 1.6 if v == 0 else 0.6, cap="butt")
        text(x0 - 6, y + 4, f"{v}", 10.5, DIM_TEXT, anchor="end")
    for i in range(N + 1):
        x = x0 + i * CELL
        line((x, PLOT_TOP), (x, PLOT_TOP + PLOT_H), GRID, 0.4, cap="butt")
    return to_plot


SAMPLES = [i / 20.0 for i in range(N * 20 + 1)]

to_plot = plot_frame(A[0], "(c) φ along the row", "one value per node; the interface is φ = 0", -6, 6)
inside = [x for x in SAMPLES if phi(x, ROW) < 0]
rect(to_plot(inside[0], 0)[0], PLOT_TOP, (inside[-1] - inside[0]) * CELL, PLOT_H, "rgba(0,128,128,0.18)")
polyline([to_plot(x, phi(x, ROW)) for x in SAMPLES], CONTOUR, 3)
for i in range(N + 1):
    circle(to_plot(i, phi(i, ROW)), 3.4, CONTOUR if phi(i, ROW) < 0 else DIM_TEXT)
text(to_plot(8, 0)[0], to_plot(8, -4.5)[1], "inside", 13, TEXT, anchor="middle")
text(to_plot(1.6, 0)[0], to_plot(1.6, 4.8)[1], "outside", 13, TEXT, anchor="middle")

to_plot = plot_frame(B[0], "(d) d and the labels along the row", "φ₂ = −d in region 2 and +d elsewhere, for any label", -6, 6)
# Label strip under the axis.
strip_y = PLOT_TOP + PLOT_H + 8
start = SAMPLES[0]
for k in range(1, len(SAMPLES) + 1):
    if k == len(SAMPLES) or label(SAMPLES[k], ROW) != label(start, ROW):
        end = SAMPLES[k - 1] if k == len(SAMPLES) else SAMPLES[k]
        l = label(start, ROW)
        x_a, x_b = to_plot(start, 0)[0], to_plot(end, 0)[0]
        rect(x_a, strip_y, x_b - x_a, 14, region_color(l))
        text(0.5 * (x_a + x_b), strip_y + 11, f"{l}", 11, PANEL, anchor="middle", weight="bold")
        if k < len(SAMPLES):
            start = SAMPLES[k]
text(B[0] - 6, strip_y + 11, "label", 10.5, DIM_TEXT, anchor="end")
# phi_2 derived from (label, d): dashed.
polyline([to_plot(x, -distance(x, ROW) if label(x, ROW) == 2 else distance(x, ROW)) for x in SAMPLES], REGION[1], 2.2, dash="7,5")
polyline([to_plot(x, distance(x, ROW)) for x in SAMPLES], TEXT, 3)
for i in range(N + 1):
    circle(to_plot(i, distance(i, ROW)), 3.4, region_color(label(i, ROW)))
text(to_plot(4.0, 0)[0], to_plot(4.0, -2.6)[1], "φ₂", 14, REGION[1], anchor="middle", weight="bold")
text(to_plot(1.6, 0)[0], to_plot(1.6, 4.8)[1], "d ≥ 0", 13, TEXT, anchor="middle")

# Legend.
LY = HEIGHT - 26
x = MARGIN
for kind, name in [("zero", "zero layer (|φ| or d ≤ ½), the nodes that move"), ("band", "narrow band (≤ 3.5 cells)"),
                   ("contour", "interface"), ("node", "node outside the band")]:
    if kind == "zero":
        circle((x + 8, LY), 6.0, "none", VERTEX, 1.6)
    elif kind == "band":
        rect(x, LY - 7, 16, 14, BAND, GRID, 1)
    elif kind == "contour":
        line((x, LY), (x + 16, LY), CONTOUR, 3)
    else:
        circle((x + 8, LY), 2.0, GRID)
    text(x + 22, LY + 4.5, name, 12, TEXT)
    x += 22 + 6.2 * len(name) + 22

emit("</svg>")
path = Path(__file__).resolve().parent / "level-set-representations.svg"
path.write_text("\n".join(out) + "\n")
print(f"wrote {path}  (legend ends at x = {x:.0f} of {WIDTH})")

# --- Second figure: one update of a node on each representation. ---------------------------------
# A boundary at x = 2.8 between an inside (or region 2, left) and an outside (or region 1, right)
# moves 0.7 cells to the right; the node at x = 3 changes side.
out.clear()
W2, H2 = 1100, 430
B0, SHIFT, NODE = 2.8, 0.7, 3
CELL2 = 60
emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W2}" height="{H2}" viewBox="0 0 {W2} {H2}" font-family="{FONT}">')
rect(0, 0, W2, H2, BACKGROUND)


def update_panel(x0, title, subtitle):
    top, h, w = 110, 200, 6 * CELL2
    rect(x0 - 50, top - 80, w + 100, h + 160, PANEL, rx=6)
    text(x0 - 30, top - 52, title, 17, TEXT, weight="bold")
    text(x0 - 30, top - 32, subtitle, 12.5, DIM_TEXT)

    def at(x, v):
        return x0 + x * CELL2, top + (3.0 - v) / 6.0 * h

    for v in range(-3, 4):
        line(at(0, v), at(6, v), GRID, 1.6 if v == 0 else 0.6, cap="butt")
        text(x0 - 8, at(0, v)[1] + 4, f"{v}", 10.5, DIM_TEXT, anchor="end")
    for i in range(7):
        line(at(i, -3), at(i, 3), GRID, 0.4, cap="butt")
    clip = f"clip{x0}"
    emit(f'<clipPath id="{clip}"><rect x="{x0}" y="{top}" width="{w}" height="{h}"/></clipPath>')
    return at, top + h, clip


def boxed(x, y, s, color):
    rect(x - 4, y - 13, 6.6 * len(s) + 8, 18, "rgba(20,20,24,0.9)", rx=3)
    text(x, y, s, 12, color)


def clipped(clip, body):
    emit(f'<g clip-path="url(#{clip})">')
    body()
    emit("</g>")


def strip(at, y, boundary, left, right, left_color, right_color, caption):
    xb = at(boundary, 0)[0]
    rect(at(0, 0)[0], y, xb - at(0, 0)[0], 14, left_color)
    rect(xb, y, at(6, 0)[0] - xb, 14, right_color)
    text(0.5 * (at(0, 0)[0] + xb), y + 11, left, 11, PANEL, anchor="middle", weight="bold")
    text(0.5 * (xb + at(6, 0)[0]), y + 11, right, 11, PANEL, anchor="middle", weight="bold")
    text(at(0, 0)[0] - 8, y + 11, caption, 10.5, DIM_TEXT, anchor="end")


xs = [i / 20.0 for i in range(121)]
after = B0 + SHIFT

# (a) one level set: phi moves, the node's sign flips.
at, bottom, clip = update_panel(90, "(a) One level set", "φ ← φ − Δt u·∇φ; a node changes side when its sign flips")
clipped(clip, lambda: (polyline([at(x, x - B0) for x in xs], CONTOUR, 2.4), polyline([at(x, x - after) for x in xs], CONTOUR, 2.4, dash="7,5")))
line(at(NODE, NODE - B0), at(NODE, NODE - after), VERTEX, 1.8)
circle(at(NODE, NODE - B0), 5, DIM_TEXT)
circle(at(NODE, NODE - after), 5, CONTOUR)
boxed(at(3.9, 0)[0], at(0, 1.1)[1], f"before: φ = +{NODE - B0:.1f}, outside", TEXT)
boxed(at(3.9, 0)[0], at(0, -1.0)[1], f"after: φ = −{after - NODE:.1f}, inside", CONTOUR)
boxed(at(0.2, 0)[0], at(0, 2.6)[1], "before (solid), after (dashed)", CONTOUR)
strip(at, bottom + 10, B0, "inside", "outside", CONTOUR, BACKGROUND_LABEL, "before")
strip(at, bottom + 30, after, "inside", "outside", CONTOUR, BACKGROUND_LABEL, "after")

# (b) MOGAC: every candidate label's signed value moves; the two smallest pick label and distance.
at, bottom, clip = update_panel(W2 // 2 + 90, "(b) MOGAC", "each candidate φₗ moves; the node takes the smallest, d = ½|min₁ − min₂|")


def mogac_curves():
    for color, sign in ((REGION[1], 1.0), (REGION[0], -1.0)):  # phi_2 = x - b, phi_1 = b - x
        polyline([at(x, sign * (x - B0)) for x in xs], color, 2.4)
        polyline([at(x, sign * (x - after)) for x in xs], color, 2.4, dash="7,5")


clipped(clip, mogac_curves)
p2, p1 = NODE - after, after - NODE
line(at(NODE, p2), at(NODE, p1), VERTEX, 1.8)
circle(at(NODE, NODE - B0), 5, REGION[0])
circle(at(NODE, p2), 5, REGION[1])
circle(at(NODE, p1), 5, REGION[0])
boxed(at(3.9, 0)[0], at(0, 1.1)[1], f"before: label 1, d = {NODE - B0:.1f}", TEXT)
boxed(at(3.9, 0)[0], at(0, 0.45)[1], f"min₂ = φ₁ = +{p1:.1f}", REGION[0])
boxed(at(3.9, 0)[0], at(0, -0.85)[1], f"min₁ = φ₂ = −{-p2:.1f}", REGION[1])
boxed(at(3.9, 0)[0], at(0, -1.5)[1], f"after: label 2, d = {0.5 * abs(p2 - p1):.1f}", TEXT)
boxed(at(0.2, 0)[0], at(0, 2.6)[1], "φ₂", REGION[1])
boxed(at(0.2, 0)[0], at(0, -2.3)[1], "φ₁", REGION[0])
strip(at, bottom + 10, B0, "2", "1", REGION[1], REGION[0], "before")
strip(at, bottom + 30, after, "2", "1", REGION[1], REGION[0], "after")

emit("</svg>")
path = Path(__file__).resolve().parent / "level-set-update.svg"
path.write_text("\n".join(out) + "\n")
print(f"wrote {path}")
