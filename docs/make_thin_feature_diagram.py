#!/usr/bin/env python3
"""Generates docs/semi-lagrangian-thin-features.svg: how the semi-Lagrangian reconstruction signs
features thinner than two springl capsules (SpringlSignField2D / SpringlSignField3D).

Each panel is computed by the same rule the code uses, on a raster cross-section (faces placed off
the lattice, as springls almost always are): U is the min
distance to the springls, capsules (U <= r) are walls, the exterior floods in from the border,
wall samples take the sign their larger-U neighbors vote for (by decreasing U), and a wall sample
with no larger-U neighbor (a local maximum of U: the middle of a feature thinner than two
capsules) is inside. Run from anywhere: python3 docs/make_thin_feature_diagram.py
"""
import math
from collections import deque
from pathlib import Path

BACKGROUND = "#202024"
PANEL = "#141418"
TEXT = "#e6e6e6"
DIM_TEXT = "#a8a8b0"
CONTOUR = "#008080"
SPRINGL = "#9a9a9a"
PARTICLE = "#c01010"
WALL_IN, WALL_OUT = "#2f9c96", "#5c5c70"
FLOOD_OUT, UNREACHED_IN = "#26262e", "#145f5c"
FALLBACK_IN = "#8a5cc8"
U_CURVE = "#d0d0d8"
FONT = "Helvetica, Arial, sans-serif"

W, H = 30, 24          # raster, in cells
R = 1.5                # capsule radius (cells)
S = 11.0               # screen pixels per cell
PAD, HEADER, PROFILE_H, FOOTER = 14, 96, 150, 18
PW, PH = W * S, H * S
BOX_W, BOX_H = PW + 2 * PAD, HEADER + PH + 16 + PROFILE_H + FOOTER
TITLE_H = 60
WIDTH = int(4 * 20 + 3 * BOX_W)
HEIGHT = int(TITLE_H + BOX_H + 72)
out = []
emit = out.append


def text(x, y, s, size=13, color=TEXT, anchor="start", weight="normal"):
    emit(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def line(a, b, color, width, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{opacity}"' if opacity is not None else ""
    emit(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{d}{o}/>')


def rect(x, y, w, h, fill, stroke="none", width=0):
    emit(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def circle(c, r, fill):
    emit(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r}" fill="{fill}"/>')


def polyline(points, color, width, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    emit(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{d}/>')


def segment_distance(q, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    t = max(0.0, min(1.0, ((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)))
    return math.hypot(q[0] - a[0] - t * ab[0], q[1] - a[1] - t * ab[1])


def tile(a, b, length=1.0):
    """Short springl segments tiling the face from a to b."""
    n = max(1, int(round(math.hypot(b[0] - a[0], b[1] - a[1]) / length)))
    return [((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n), (a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n))
            for k in range(n)]


def box(x0, x1, y0, y1):
    """Springls all around a rectangular body (its four faces)."""
    return tile((x0, y0), (x1, y0)) + tile((x1, y0), (x1, y1)) + tile((x1, y1), (x0, y1)) + tile((x0, y1), (x0, y0))


def reconstruct(springls):
    """The code's rule on a raster: U, walls, exterior flood, upwind sweep, inside fallback."""
    U = {(i, j): min(segment_distance((i, j), a, b) for a, b in springls) for i in range(W + 1) for j in range(H + 1)}
    wall = {k: U[k] <= R for k in U}

    def nb(k):
        i, j = k
        for m in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if m in U:
                yield m

    queue = deque(k for k in U if (k[0] in (0, W) or k[1] in (0, H)) and not wall[k])
    exterior = set(queue)
    while queue:
        for m in nb(queue.popleft()):
            if not wall[m] and m not in exterior:
                exterior.add(m)
                queue.append(m)
    sign, kind = {}, {}
    for k in U:
        if not wall[k]:
            sign[k] = 1 if k in exterior else -1
            kind[k] = "flood" if k in exterior else "unreached"
    for k in sorted((k for k in U if wall[k]), key=lambda k: (-U[k], k)):
        vote, upwind = 0.0, False
        for m in nb(k):
            if m in sign and U[m] > U[k]:
                vote += sign[m] * (U[m] - U[k])
                upwind = True
        if upwind and vote != 0.0:
            sign[k], kind[k] = (-1 if vote < 0 else 1), "wall"
        else:
            sign[k], kind[k] = -1, "fallback"
    return U, sign, kind


def zero_crossings_y(U, sign, i):
    """y where phi = sign * U changes sign along column i (linear interpolation)."""
    ys = []
    for j in range(H):
        a, b = sign[i, j] * U[i, j], sign[i, j + 1] * U[i, j + 1]
        if (a < 0) != (b < 0):
            ys.append(j + a / (a - b))
    return ys


cases = [
    ("(a) Thick sheet", ["Walls on each face; the flood stops at them", "and never reaches the interior: inside. The", "fallback decides only the odd corner sample."],
     box(2.3, 27.7, 7.3, 15.3), [(2.3, 27.7, 7.3, 15.3)]),
    ("(b) Sheet thinner than two capsules", ["The two faces' walls merge: no interior is left", "to be unreached. Its middle is a local maximum", "of U with no upwind neighbor: inside (fallback)."],
     box(2.3, 27.7, 11.3, 12.9), [(2.3, 27.7, 11.3, 12.9)]),
    ("(c) Outside gap thinner than two capsules", ["The same local maximum of U lies between two", "bodies; the fallback makes it inside, so they", "fuse: a union, the price of the rule."],
     box(2.3, 27.7, 2.3, 11.3) + box(2.3, 27.7, 12.9, 21.7), [(2.3, 27.7, 2.3, 11.3), (2.3, 27.7, 12.9, 21.7)]),
]

emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}" role="img" aria-labelledby="title">')
emit('<title id="title">How the semi-Lagrangian reconstruction signs features thinner than two springl capsules</title>')
emit(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>')
text(20, 34, "Features thinner than two capsules: the wall fallback", 22, TEXT, weight="bold")
text(WIDTH - 20, 34, "a wall sample with no larger-U neighbor is inside", 15, DIM_TEXT, "end")
colors = {"flood": FLOOD_OUT, "unreached": UNREACHED_IN, "fallback": FALLBACK_IN}

for c, (title, lines, springls, truth) in enumerate(cases):
    U, sign, kind = reconstruct(springls)
    x0, y0 = 20 + c * (BOX_W + 20), TITLE_H
    rect(x0, y0, BOX_W, BOX_H, PANEL)
    text(x0 + PAD, y0 + 24, title, 16, TEXT, weight="bold")
    for k, s in enumerate(lines):
        text(x0 + PAD, y0 + 46 + 16 * k, s, 12, DIM_TEXT)
    ox, oy = x0 + PAD, y0 + HEADER

    def P(p):
        return (ox + p[0] * S, oy + (H - p[1]) * S)

    # Raster samples as cells.
    for (i, j), k in kind.items():
        color = colors.get(k) or (WALL_IN if sign[i, j] < 0 else WALL_OUT)
        a = P((i - 0.5, j + 0.5))
        rect(a[0], a[1], S, S, color)
    for i in range(W + 2):
        line(P((i - 0.5, -0.5)), P((i - 0.5, H + 0.5)), "#000000", 0.5, opacity=0.35)
    for j in range(H + 2):
        line(P((-0.5, j - 0.5)), P((W + 0.5, j - 0.5)), "#000000", 0.5, opacity=0.35)
    # The true material (dashed) and the springls.
    for bx0, bx1, lo, hi in truth:
        a, b = P((bx0, hi)), P((bx1, lo))
        emit(f'<rect x="{a[0]:.1f}" y="{a[1]:.1f}" width="{b[0] - a[0]:.1f}" height="{b[1] - a[1]:.1f}" fill="none" stroke="{TEXT}" stroke-width="1.2" stroke-dasharray="5,4"/>')
    for a, b in springls:
        line(P(a), P(b), SPRINGL, 2.0)
        circle(P(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)), 2.2, PARTICLE)
    # The reconstructed contour, along the middle column.
    for i in range(W + 1):
        for yz in zero_crossings_y(U, sign, i):
            circle(P((i, yz)), 1.6, "#2fd0c8")
    # Profile of U across the feature (column W/2), with the wall threshold and the sign.
    i = W // 2
    line(P((i, -0.5)), P((i, H + 0.5)), U_CURVE, 1.0, dash="3,3")
    py = oy + PH + 16
    rect(ox, py, PW, PROFILE_H, "#1a1a20")
    umax = max(U[i, j] for j in range(H + 1))
    scale_u = (PROFILE_H - 30) / max(4.0, min(umax, 8.0))
    base = py + PROFILE_H - 16

    def Q(j, u):
        return (ox + (j / H) * PW, base - min(u, 8.0) * scale_u)

    for j in range(H):
        color = colors.get(kind[i, j]) or (WALL_IN if sign[i, j] < 0 else WALL_OUT)
        a = Q(j, 0)
        rect(a[0], base + 3, PW / H, 8, color)
    line(Q(0, R), Q(H, R), "#e0b040", 1.0, dash="5,3")
    text(ox + PW - 4, Q(H, R)[1] - 4, "r: wall", 11, "#e0b040", "end")
    polyline([Q(j, U[i, j]) for j in range(H + 1)], U_CURVE, 1.6)
    for j in range(1, H):
        if U[i, j] <= R and U[i, j] > U[i, j - 1] and U[i, j] >= U[i, j + 1] and kind[i, j] == "fallback":
            circle(Q(j, U[i, j]), 4, FALLBACK_IN)
            text(Q(j, U[i, j])[0] + 7, Q(j, U[i, j])[1] - 6, "local max of U: inside", 11, TEXT)
    text(ox + 4, py + 14, f"U along the dashed column (bottom to top), and the sign", 11, DIM_TEXT)

# Legend.
ly = HEIGHT - 26
x = 20
for color, label in [(FLOOD_OUT, "flood: outside"), (UNREACHED_IN, "unreached: inside"), (WALL_OUT, "wall, upwind out"), (WALL_IN, "wall, upwind in"),
                     (FALLBACK_IN, "wall, no upwind neighbor: inside")]:
    rect(x, ly - 9, 13, 13, color, "#55555f", 0.8)
    text(x + 19, ly + 2, label, 12)
    x += 19 + 6.6 * len(label) + 18
line((x, ly - 3), (x + 18, ly - 3), TEXT, 1.2, dash="5,4")
text(x + 24, ly + 2, "true material", 12)
x += 24 + 6.6 * 13 + 18
circle((x + 6, ly - 3), 3, "#2fd0c8")
text(x + 16, ly + 2, "reconstructed contour", 12)

emit("</svg>")
path = Path(__file__).resolve().parent / "semi-lagrangian-thin-features.svg"
path.write_text("\n".join(out) + "\n")
print(f"wrote {path} ({WIDTH} x {HEIGHT})")
