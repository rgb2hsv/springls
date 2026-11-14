#!/usr/bin/env python3
"""Generates docs/semi-lagrangian-reconstruction.svg: how semi-Lagrangian doubleweave rebuilds the
level set from its springls after a sub-step (SemiLagrangianDoubleweaveLevelSet2D +
SpringlSignField2D).

The panels are computed, not drawn by hand: a small re-implementation of the reconstruction (min
unsigned distance to the springls, capsules as walls, exterior flood from the border, upwind sign
sweep into the walls) runs on a doubleweave blob that was advected 8 cells in one sub-step and has
a gap in its primal constellation, plus a small blob.
Colors match the enright2d panels. Run from anywhere: python3 docs/make_semi_lagrangian_diagram.py
"""
import math
from collections import deque
from pathlib import Path

# Palette (same as examples/enright2d/Renderer.cpp and docs/make_anatomy_diagram.py).
BACKGROUND = "#202024"
PANEL = "#141418"
TEXT = "#e6e6e6"
DIM_TEXT = "#a8a8b0"
CONTOUR = "#008080"
PRIMAL_SPRINGL = "#9a9a9a"
PRIMAL_PARTICLE = "#c01010"
DUAL_SPRINGL = "#ff8c00"
DUAL_PARTICLE = "#2a85ff"
VERTEX = "#ffcc00"
HIGHLIGHT = "#ffffff"
TRACE = "#ffcc00"
PRIMAL_TINT = (236, 64, 122)
DUAL_TINT = (64, 160, 236)
GRID_DEPTH = ["#464652", "#3c698c", "#379691", "#c8a53c"]
# Sign raster classes.
WALL_IN, WALL_OUT = "#2f9c96", "#5c5c70"
FLOOD_OUT, UNREACHED_IN = "#26262e", "#145f5c"
FALLBACK_IN = "#8a5cc8"
FONT = "Helvetica, Arial, sans-serif"

# ---- Scene (world units = finest grid cells, y up) -----------------------------------------------
W, H = 64, 48             # domain
SPACING = 1.0             # sign raster spacing = finest cell size h
R_BAND = 1.5              # capsule radius r = capsuleCells * h
BAND_WIDTH = 1.5          # phi clamp (drawn in cells)
DELTA = (8.0, 0.0)        # displacement of the flow over this sub-step
C_NEW = (36.0, 22.0)
C_OLD = (C_NEW[0] - DELTA[0], C_NEW[1] - DELTA[1])
HOLE = (-0.95, -0.55)     # particle angles (rad, about C_NEW) with no primal springls (the dual covers them)
SMALL = ((56.0, 39.0), 4.2)  # a second, intact blob


def radius(theta):
    return 12.0 + 2.2 * math.cos(3 * theta + 0.5)


def blob_points(center, n=44):
    return [(center[0] + radius(2 * math.pi * k / n) * math.cos(2 * math.pi * k / n),
             center[1] + radius(2 * math.pi * k / n) * math.sin(2 * math.pi * k / n)) for k in range(n)]


def circle_points(center, r, n=16):
    return [(center[0] + r * math.cos(2 * math.pi * k / n), center[1] + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def inside_blob(p, center):
    dx, dy = p[0] - center[0], p[1] - center[1]
    return math.hypot(dx, dy) < radius(math.atan2(dy, dx))


def sub(a, b): return (a[0] - b[0], a[1] - b[1])
def add(a, b): return (a[0] + b[0], a[1] + b[1])
def mul(a, s): return (a[0] * s, a[1] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1]
def mid(a, b): return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def make_springls(vertexes, hole_center=None):
    """Doubleweave springls on a CCW polygon: primal on edges, dual chords on vertexes."""
    springls = []  # (v0, v1, particle, normal, constellation)
    n = len(vertexes)

    def normal(v0, v1):
        t = sub(v1, v0)
        length = math.hypot(*t)
        return (t[1] / length, -t[0] / length)  # outward for a CCW polygon

    def in_hole(p):
        if hole_center is None:
            return False
        a = math.atan2(p[1] - hole_center[1], p[0] - hole_center[0])
        return HOLE[0] <= a <= HOLE[1]

    for k in range(n):
        a, b = vertexes[k], vertexes[(k + 1) % n]
        p = mid(a, b)
        if not in_hole(p):
            springls.append((a, b, p, normal(a, b), 0))
    for k in range(n):
        u, v, w = vertexes[k - 1], vertexes[k], vertexes[(k + 1) % n]
        m0, m1 = mid(u, v), mid(v, w)
        shift = sub(v, mid(m0, m1))
        a, b = add(m0, shift), add(m1, shift)
        springls.append((a, b, v, normal(a, b), 1))  # the gap is in the primal constellation only
    return springls


SPRINGLS = make_springls(blob_points(C_NEW), C_NEW) + make_springls(circle_points(*SMALL))


def closest_on(s, q):
    a, b = s[0], s[1]
    ab = sub(b, a)
    t = max(0.0, min(1.0, dot(sub(q, a), ab) / dot(ab, ab)))
    c = add(a, mul(ab, t))
    return c, t, math.hypot(q[0] - c[0], q[1] - c[1])


def nearest(q):
    best = None
    for i, s in enumerate(SPRINGLS):
        c, t, d = closest_on(s, q)
        if best is None or d < best[3]:
            best = (i, c, t, d)
    return best


# ---- Reconstruction on the raster ----------------------------------------------------------------
NX, NY = int(W / SPACING) + 1, int(H / SPACING) + 1


def node(i, j): return (i * SPACING, j * SPACING)


U = {}
WALL = {}
SIGN = {}
CLASS = {}
for j in range(NY):
    for i in range(NX):
        U[i, j] = nearest(node(i, j))[3]
        WALL[i, j] = U[i, j] <= R_BAND


def neighbors4(i, j):
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        m = (i + di, j + dj)
        if 0 <= m[0] < NX and 0 <= m[1] < NY:
            yield m


# The exterior floods in from the border around the walls; what it does not reach is inside.
queue = deque((i, j) for i in range(NX) for j in range(NY) if (i in (0, NX - 1) or j in (0, NY - 1)) and not WALL[i, j])
exterior = set(queue)
while queue:
    for m in neighbors4(*queue.popleft()):
        if not WALL[m] and m not in exterior:
            exterior.add(m)
            queue.append(m)
for k in U:
    if not WALL[k]:
        SIGN[k] = 1 if k in exterior else -1
        CLASS[k] = "flood" if k in exterior else "unreached"
# Upwind into the walls, by decreasing U: each takes its larger-U neighbors' sign, weighted by how
# much larger; a local maximum of U (no upwind neighbor) is inside.
for k in sorted((k for k in U if WALL[k]), key=lambda k: (-U[k], k)):
    vote, any_upwind = 0.0, False
    for m in neighbors4(*k):
        if m in SIGN and U[m] > U[k]:
            vote += SIGN[m] * (U[m] - U[k])
            any_upwind = True
    if any_upwind and vote != 0.0:
        SIGN[k] = -1 if vote < 0 else 1
        CLASS[k] = "wall"
    else:
        SIGN[k] = -1
        CLASS[k] = "fallback"

PHI = {k: SIGN[k] * min(U[k], BAND_WIDTH) for k in U}

# ---- SVG helpers ---------------------------------------------------------------------------------
out = []
emit = out.append


def text(x, y, s, size=14, color=TEXT, anchor="start", weight="normal", italic=False):
    style = ' font-style="italic"' if italic else ""
    emit(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{style}>{s}</text>')


def line(a, b, color, width, dash=None, opacity=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{opacity}"' if opacity is not None else ""
    emit(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="{cap}"{d}{o}/>')


def circle(c, r, fill, stroke=None, width=0):
    s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    emit(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r}" fill="{fill}"{s}/>')


def polygon(points, fill, stroke="none", width=0, dash=None, fill_opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    fo = f' fill-opacity="{fill_opacity}"' if fill_opacity is not None else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    emit(f'<polygon points="{pts}" fill="{fill}"{fo} stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"{d}/>')


def arrow(a, b, color, width=2.0, head=8, dash=None):
    line(a, b, color, width, dash)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45))
    p2 = (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))
    emit(f'<polyline points="{p1[0]:.1f},{p1[1]:.1f} {b[0]:.1f},{b[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="none" stroke="{color}" '
         f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')


def hex_rgb(rgb): return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v)))) for v in rgb)


# ---- Layout ----------------------------------------------------------------------------------------
S = 8.0                         # screen pixels per cell
PW, PH = W * S, H * S           # plot size
HEADER, FOOTER, PAD = 84, 46, 14
BOX_W, BOX_H = PW + 2 * PAD, HEADER + PH + FOOTER
TITLE_H = 58
WIDTH = int(3 * 20 + 2 * BOX_W)
HEIGHT = int(TITLE_H + 2 * BOX_H + 3 * 16)


class Panel:
    def __init__(self, col, row, title, lines):
        self.x = 20 + col * (BOX_W + 20)
        self.y = TITLE_H + row * (BOX_H + 16)
        self.ox, self.oy = self.x + PAD, self.y + HEADER
        emit(f'<rect x="{self.x}" y="{self.y}" width="{BOX_W}" height="{BOX_H}" rx="6" fill="{PANEL}"/>')
        text(self.x + PAD, self.y + 24, title, 17, TEXT, weight="bold")
        for k, s in enumerate(lines):
            text(self.x + PAD, self.y + 44 + 16 * k, s, 12.5, DIM_TEXT)
        emit(f'<clipPath id="clip{col}{row}"><rect x="{self.ox}" y="{self.oy}" width="{PW}" height="{PH}"/></clipPath>')
        self.clip = f"clip{col}{row}"

    def __call__(self, p):
        return (self.ox + p[0] * S, self.oy + (H - p[1]) * S)

    def begin_clip(self):
        emit(f'<g clip-path="url(#{self.clip})">')

    def end_clip(self):
        emit("</g>")
        emit(f'<rect x="{self.ox}" y="{self.oy}" width="{PW}" height="{PH}" fill="none" stroke="#33333c" stroke-width="1"/>')

    def cells(self, color_of, step=SPACING):
        """Fills one square per raster node (centered on it), merging horizontal runs."""
        for j in range(NY):
            i = 0
            while i < NX:
                color = color_of(i, j)
                k = i
                while k + 1 < NX and color_of(k + 1, j) == color:
                    k += 1
                if color:
                    x0, y0 = self(((i - 0.5) * step, (j + 0.5) * step))
                    emit(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{(k - i + 1) * step * S:.1f}" height="{step * S:.1f}" fill="{color}"/>')
                i = k + 1

    def legend(self, items):
        x, y = self.x + PAD, self.y + BOX_H - 20
        for kind, color, label in items:
            if kind == "swatch":
                emit(f'<rect x="{x}" y="{y - 9}" width="13" height="13" fill="{color}" stroke="#55555f" stroke-width="0.8"/>')
                x += 19
            elif kind == "line":
                line((x, y - 2.5), (x + 18, y - 2.5), color, 3)
                x += 24
            elif kind == "dash":
                line((x, y - 2.5), (x + 18, y - 2.5), color, 1.6, dash="4,3", cap="butt")
                x += 24
            elif kind == "arrow":
                arrow((x, y - 2.5), (x + 18, y - 2.5), color, 1.8, 6)
                x += 24
            text(x, y + 2, label, 12, TEXT)
            x += 6.3 * len(label) + 16


def draw_springls(panel, faint=False, particles=True):
    opacity = 0.45 if faint else 1.0
    for v0, v1, p, n, c in SPRINGLS:
        color = DUAL_SPRINGL if c else PRIMAL_SPRINGL
        line(panel(v0), panel(v1), color, 2.2 if not faint else 1.6, opacity=opacity)
    if particles and not faint:
        for v0, v1, p, n, c in SPRINGLS:
            circle(panel(p), 2.2, DUAL_PARTICLE if c else PRIMAL_PARTICLE)


def marching_squares(values, level=0.0):
    """Segments of {values = level} on the node grid (linear interpolation)."""
    segs = []
    for j in range(NY - 1):
        for i in range(NX - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            v = [values[c] - level for c in corners]
            pts = []
            for e in range(4):
                a, b = corners[e], corners[(e + 1) % 4]
                va, vb = v[e], v[(e + 1) % 4]
                if (va < 0) != (vb < 0):
                    t = va / (va - vb)
                    pts.append(((a[0] + t * (b[0] - a[0])) * SPACING, (a[1] + t * (b[1] - a[1])) * SPACING))
            if len(pts) == 2:
                segs.append((pts[0], pts[1]))
            elif len(pts) == 4:
                center = sum(v) / 4
                if (center < 0) == (v[0] < 0):
                    segs += [(pts[0], pts[1]), (pts[2], pts[3])]
                else:
                    segs += [(pts[0], pts[3]), (pts[1], pts[2])]
    return segs


def inside_polygons(values):
    """Per-cell polygons of {values < 0} (each square clipped by linear interpolation)."""
    polys = []
    for j in range(NY - 1):
        for i in range(NX - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            v = [values[c] for c in corners]
            if all(x >= 0 for x in v):
                continue
            pts = []
            for e in range(4):
                a, b = corners[e], corners[(e + 1) % 4]
                va, vb = v[e], v[(e + 1) % 4]
                if va < 0:
                    pts.append((a[0] * SPACING, a[1] * SPACING))
                if (va < 0) != (vb < 0):
                    t = va / (va - vb)
                    pts.append(((a[0] + t * (b[0] - a[0])) * SPACING, (a[1] + t * (b[1] - a[1])) * SPACING))
            polys.append(pts)
    return polys


# ---- Document --------------------------------------------------------------------------------------
emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
     f'font-family="{FONT}" role="img" aria-labelledby="title">')
emit('<title id="title">Semi-Lagrangian doubleweave: the level set rebuilt from the springls after one sub-step</title>')
emit(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>')
text(20, 34, "Semi-Lagrangian doubleweave: one sub-step", 22, TEXT, weight="bold")
text(WIDTH - 20, 34, "&#966;(x) = s(x) &#183; min(U(x), band)   on every grid, after every sub-step", 15, DIM_TEXT, anchor="end", italic=True)

old_outline = blob_points(C_OLD, 120)
new_outline = blob_points(C_NEW, 120)
hole_mid = sum(HOLE) / 2
hole_point = (C_NEW[0] + radius(hole_mid) * math.cos(hole_mid), C_NEW[1] + radius(hole_mid) * math.sin(hole_mid))

# (1) Advect freely ------------------------------------------------------------------------------
p1 = Panel(0, 0, "1. Advect the springls freely", [
    "RK4 moves every springl 8 cells in one sub-step. A level set evolved by an upwind",
    "PDE could follow at most half a cell per update (the CFL limit). This level set is",
    "never evolved: it is rebuilt from the springls after the sub-step."])
p1.begin_clip()
polygon([p1(p) for p in old_outline], "none", DIM_TEXT, 1.6, dash="6,4")
for k in range(0, 44, 6):
    a = 2 * math.pi * k / 44
    old = (C_OLD[0] + radius(a) * math.cos(a), C_OLD[1] + radius(a) * math.sin(a))
    arrow(p1(old), p1(add(old, mul(DELTA, 0.93))), "#7fd17f", 1.5, 7)
draw_springls(p1)
p1.end_clip()
hp = p1(hole_point)
line((hp[0] + 34, hp[1] + 26), (hp[0] + 6, hp[1] + 4), DIM_TEXT, 1)
text(hp[0] + 37, hp[1] + 31, "gap in the primal", 12, TEXT)
text(hp[0] + 37, hp[1] + 46, "springls; the dual covers it", 12, TEXT)
p1.legend([("dash", DIM_TEXT, "before the sub-step"), ("arrow", "#7fd17f", "flow"),
           ("line", PRIMAL_SPRINGL, "primal"), ("line", DUAL_SPRINGL, "dual")])

# (2) Local distance field -----------------------------------------------------------------------
p2 = Panel(1, 0, "2. Unsigned distance U; the capsules are walls", [
    "U = min over the springls of both constellations: the distance to the union of",
    "their capsules. Nodes with U &#8804; r (1.5 cells) are walls. No springl normals are",
    "used. Primal and dual overlap, so the walls close over one constellation's gaps."])
p2.begin_clip()
FINE = 0.5
fine_nx, fine_ny = int(W / FINE) + 1, int(H / FINE) + 1
LEVELS = 6
for j in range(fine_ny):
    row = []
    for i in range(fine_nx):
        q = (i * FINE, j * FINE)
        hit = nearest(q)
        if hit[3] > R_BAND:
            row.append(None)
            continue
        t = math.floor(hit[3] / R_BAND * LEVELS) / LEVELS
        tint = DUAL_TINT if SPRINGLS[hit[0]][4] else PRIMAL_TINT
        bg = (20, 20, 24)
        w = 0.85 * (1 - t) + 0.12
        row.append(hex_rgb(tuple(bg[k] + w * (tint[k] - bg[k]) for k in range(3))))
    i = 0
    while i < fine_nx:
        k = i
        while k + 1 < fine_nx and row[k + 1] == row[i]:
            k += 1
        if row[i]:
            x0, y0 = p2(((i - 0.5) * FINE, (j + 0.5) * FINE))
            emit(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{(k - i + 1) * FINE * S:.2f}" height="{FINE * S:.2f}" fill="{row[i]}"/>')
        i = k + 1
for a, b in marching_squares(U, R_BAND):
    line(p2(a), p2(b), "#d0d0d8", 1.1, opacity=0.8)
draw_springls(p2)
p2.end_clip()
hp = p2(hole_point)
line((hp[0] + 34, hp[1] + 26), (hp[0] + 8, hp[1] + 4), DIM_TEXT, 1)
text(hp[0] + 37, hp[1] + 31, "dual capsules alone:", 12, TEXT)
text(hp[0] + 37, hp[1] + 46, "the wall stays closed", 12, TEXT)
p2.legend([("swatch", hex_rgb(PRIMAL_TINT), "nearest: primal"), ("swatch", hex_rgb(DUAL_TINT), "nearest: dual"),
           ("line", "#d0d0d8", "wall edge U = r")])

# (3) Sign field ---------------------------------------------------------------------------------
p3 = Panel(0, 1, "3. Sign from topology: flood, then upwind", [
    "The exterior floods in from the border around the walls; what it cannot reach is",
    "inside. Wall nodes, by decreasing U, take the sign their larger-U neighbors vote for,",
    "so outside and inside meet at U = 0, on the springls. No previous field is read."])
colors = {("wall", -1): WALL_IN, ("wall", 1): WALL_OUT, ("flood", 1): FLOOD_OUT, ("unreached", -1): UNREACHED_IN,
          ("fallback", -1): FALLBACK_IN}
p3.begin_clip()
p3.cells(lambda i, j: colors[CLASS[i, j], SIGN[i, j]])
# Raster cell lines.
for i in range(NX + 1):
    line(p3(((i - 0.5) * SPACING, -0.5)), p3(((i - 0.5) * SPACING, H + 0.5)), "#000000", 0.5, opacity=0.35, cap="butt")
for j in range(NY + 1):
    line(p3((-0.5, (j - 0.5) * SPACING)), p3((W + 0.5, (j - 0.5) * SPACING)), "#000000", 0.5, opacity=0.35, cap="butt")
draw_springls(p3, faint=True)
# The flood, from the border inward.
for start, end in [((4, 44), (12, 36)), ((60, 4), (52, 10)), ((4, 4), (14, 10))]:
    arrow(p3(start), p3(end), "#d0d0d8", 1.8, 8)
p3.end_clip()
sp = p3((SMALL[0][0] - SMALL[1] - 2.5, SMALL[0][1]))
text(sp[0], sp[1] - 2, "not reached by", 12, TEXT, anchor="end")
text(sp[0], sp[1] + 13, "the flood: inside", 12, TEXT, anchor="end")
p3.legend([("swatch", FLOOD_OUT, "flood: outside"), ("swatch", UNREACHED_IN, "unreached: inside"), ("swatch", WALL_OUT, "wall, upwind out"),
           ("swatch", WALL_IN, "wall, upwind in")])

# (4) Level set on the adaptive grid --------------------------------------------------------------
p4 = Panel(1, 1, "4. &#966; = s &#183; min(U, band) on the adaptive grid, and its contour", [
    "Leaves near the springls are at the finest depth; coarse leaves hold &#177;band.",
    "The contour runs along the springls, closed across the primal gap, where fill",
    "seeds new primal springls in the same sub-step."])
p4.begin_clip()
# Quadtree: base cells of 8, refined to 1 within the band margin of a particle.
particles = [s[2] for s in SPRINGLS]
MARGIN = 2.5


def near(x0, y0, size):
    return any(x0 - MARGIN <= p[0] <= x0 + size + MARGIN and y0 - MARGIN <= p[1] <= y0 + size + MARGIN for p in particles)


leaves = []


def refine(x0, y0, size, depth):
    if depth < 3 and near(x0, y0, size):
        half = size / 2
        for dx in (0, half):
            for dy in (0, half):
                refine(x0 + dx, y0 + dy, half, depth + 1)
    else:
        leaves.append((x0, y0, size, depth))


for bx in range(0, W, 8):
    for by in range(0, H, 8):
        refine(bx, by, 8, 0)
for depth in range(4):
    for x0, y0, size, d in leaves:
        if d != depth:
            continue
        a, b = p4((x0, y0 + size)), p4((x0 + size, y0))
        emit(f'<rect x="{a[0]:.1f}" y="{a[1]:.1f}" width="{b[0] - a[0]:.1f}" height="{b[1] - a[1]:.1f}" fill="none" '
             f'stroke="{GRID_DEPTH[depth]}" stroke-width="{0.7 if depth == 3 else 1.0}"/>')
for poly in inside_polygons(PHI):
    polygon([p4(p) for p in poly], CONTOUR, fill_opacity=0.45)
draw_springls(p4, faint=True)
for a, b in marching_squares(PHI):
    line(p4(a), p4(b), CONTOUR, 2.6)
p4.end_clip()
hp = p4(hole_point)
line((hp[0] + 34, hp[1] + 26), (hp[0] + 9, hp[1] + 5), DIM_TEXT, 1)
text(hp[0] + 37, hp[1] + 31, "closed across the gap;", 12, TEXT)
text(hp[0] + 37, hp[1] + 46, "fill re-seeds primal here", 12, TEXT)
p4.legend([("line", CONTOUR, "contour"),
           ("swatch", GRID_DEPTH[0], "depth 0"), ("swatch", GRID_DEPTH[1], "1"), ("swatch", GRID_DEPTH[2], "2"),
           ("swatch", GRID_DEPTH[3], "3 (finest)")])

emit("</svg>")
path = Path(__file__).resolve().parent / "semi-lagrangian-reconstruction.svg"
path.write_text("\n".join(out) + "\n")
counts = {}
for v in CLASS.values():
    counts[v] = counts.get(v, 0) + 1
print(f"wrote {path} ({WIDTH} x {HEIGHT}); raster classes {counts}")
