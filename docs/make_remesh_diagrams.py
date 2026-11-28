#!/usr/bin/env python3
"""Generates the mesh tracking figures of the documentation (section 3.15):

- docs/remesh-operations.svg: the three quality operations of AdaptiveRemesh3D on small patches, before and
  after: an edge split at its midpoint, an edge collapse to its midpoint, and an edge flip that raises the
  smallest angle of its two triangles.
- docs/remesh-parallel.svg: how the operations run in parallel. (a) Independent sets: every long edge is a
  candidate, candidates claim their triangles longest first (an atomic minimum), and every candidate that
  holds all its claims is applied at once. (b) Charts: each chart is remeshed on its own thread with the ring
  of triangles around it frozen. (c) Then the band of triangles along the chart borders is remeshed by
  independent sets, the rest frozen.

The panels are computed, not drawn by hand: small 2D re-implementations of the operations
(AdaptiveRemesh3D; the charts are a 2D Lloyd stand-in for CvtCharts). Colors are the toybox palette (apps/windows/common/Palette.hpp).
Run from anywhere: python3 docs/make_remesh_diagrams.py
"""
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent

BACKGROUND = "#1E1F24"
PANEL = "#15161A"
TEXT = "#E6E6E6"
DIM_TEXT = "#A8A8B0"
WIRE = "#8A93A6"
FACE = "#2A2D36"
CONTOUR = "#3FC1C9"
VERTEX = "#F5C542"
HIGHLIGHT = "#E048B8"
NEW = "#7BD88F"
LOSER = "#E3B65C"
MARKED = "#D23A48"
FROZEN = "#4A4E5C"
CHARTS = ["#3E6F94", "#5560AA", "#7D5AA6", "#A65A9A", "#3F8F7A", "#8C7A3E", "#4F7F9F", "#6E5B8F"]
CURVE_A = "#3FC1C9"
CURVE_B = "#F2EBD3"
FONT = "Inter, Helvetica, Arial, sans-serif"


def sub(a, b): return (a[0] - b[0], a[1] - b[1])
def add(a, b): return (a[0] + b[0], a[1] + b[1])
def mul(a, s): return (a[0] * s, a[1] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1]
def norm(a): return math.hypot(a[0], a[1])
def mid(a, b): return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
def cross(a, b): return a[0] * b[1] - a[1] * b[0]


def angle_at(p, a, b):
    u, v = sub(a, p), sub(b, p)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(u, v) / (norm(u) * norm(v))))))


def min_angle(t):
    a, b, c = t
    return min(angle_at(a, b, c), angle_at(b, c, a), angle_at(c, a, b))


class Svg:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.parts = [f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>']

    def add(self, s):
        self.parts.append(s)

    def panel(self, x, y, w, h, title):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PANEL}"/>')
        self.text(x + 14, y + 26, title, size=15, weight=600)

    def text(self, x, y, s, size=13, color=TEXT, anchor="start", weight=400, halo=False):
        style = f' stroke="{PANEL}" stroke-width="3.5" stroke-linejoin="round" paint-order="stroke"' if halo else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{style}>{s}</text>')

    def line(self, a, b, color, width=1.5, dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{color}" '
                 f'stroke-width="{width}" stroke-linecap="round" opacity="{opacity}"{d}/>')

    def polygon(self, pts, fill, opacity=1.0, stroke="none", width=0.0):
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<polygon points="{p}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" '
                 f'stroke-width="{width}" stroke-linejoin="round"/>')

    def circle(self, c, r, fill, stroke="none", width=0.0):
        self.add(f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def arrow(self, a, b, color, width=2.0, head=9.0):
        d = sub(b, a)
        u = mul(d, 1.0 / norm(d))
        n = (-u[1], u[0])
        base = sub(b, mul(u, head))
        self.line(a, base, color, width)
        self.polygon([b, add(base, mul(n, head * 0.42)), sub(base, mul(n, head * 0.42))], color)

    def save(self, name):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
               f'viewBox="0 0 {self.width} {self.height}">\n{body}\n</svg>\n')
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote {OUT / name}")


def fit(points, box, margin=0.08):
    """World (y up) to screen, uniformly scaled into box = (x, y, w, h)."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bx, by, bw, bh = box
    s = min(bw * (1 - 2 * margin) / max(x1 - x0, 1e-9), bh * (1 - 2 * margin) / max(y1 - y0, 1e-9))
    ox = bx + (bw - s * (x1 - x0)) / 2
    oy = by + (bh - s * (y1 - y0)) / 2
    return lambda p: (ox + s * (p[0] - x0), oy + s * (y1 - p[1]))


def draw_mesh(svg, P, T, f, fill=FACE, wire=WIRE, width=1.2, faces=None):
    for t in T:
        svg.polygon([f(P[i]) for i in t], (faces or {}).get(tuple(t), fill), stroke=wire, width=width)


# ---- Figure: the three operations ---------------------------------------------------------------------


def figure_operations():
    W, H = 1140, 400
    svg = Svg(W, H)
    pw, ph, gap = 364, 344, 14
    titles = ["(a) Split: an edge longer than 1.5 L", "(b) Collapse: an edge shorter than 0.5 L", "(c) Flip: the smallest angle rises"]
    for k in range(3):
        x = 14 + k * (pw + gap)
        svg.panel(x, 14, pw, ph, titles[k])

    # (a) split: a hinge whose shared edge pq is long.
    p, q, c, d = (0.0, 0.0), (3.4, 0.25), (1.6, 1.15), (1.85, -1.05)
    before = [(p, q, c), (q, p, d)]
    m = mid(p, q)
    after = [(p, m, c), (m, q, c), (q, m, d), (m, p, d)]
    draw_pair(svg, 14, before, after, [(p, q)], [m], [(m, c), (m, d)],
              f"|pq| = {norm(sub(q, p)):.2f} L", "midpoint m, 4 triangles")

    # (b) collapse: a short edge ab inside a fan of eight ring vertexes.
    a, b = (-0.17, 0.02), (0.17, -0.02)
    ring = []
    for i in range(8):
        th = 2 * math.pi * (i + 0.5) / 8
        r = 1.15 + 0.12 * math.sin(3 * i)
        ring.append((r * math.cos(th), r * math.sin(th)))
    side = ["b" if pt[0] > 0 else "a" for pt in ring]
    tris = []
    for i in range(8):
        r0, r1 = ring[i], ring[(i + 1) % 8]
        s0, s1 = side[i], side[(i + 1) % 8]
        c0 = a if s0 == "a" else b
        tris.append((c0, r0, r1))
        if s0 != s1:
            c1 = b if s0 == "a" else a
            tris.append((c0, r1, c1))
    mm = mid(a, b)
    after_b = []
    for t in tris:
        if a in t and b in t:
            continue  # the two triangles of ab vanish
        after_b.append(tuple(mm if v in (a, b) else v for v in t))
    draw_pair(svg, 14 + pw + gap, tris, after_b, [(a, b)], [mm], [],
              f"|ab| = {norm(sub(b, a)):.2f} L", "a, b merged at the midpoint")

    # (c) flip: two slivers sharing a long diagonal.
    A, B, C, D = (0.0, 0.0), (1.35, -0.62), (2.9, 0.05), (1.5, 0.66)
    before_c = [(A, B, C), (A, C, D)]
    after_c = [(A, B, D), (B, C, D)]
    m0 = min(min_angle(t) for t in before_c)
    m1 = min(min_angle(t) for t in after_c)
    draw_pair(svg, 14 + 2 * (pw + gap), before_c, after_c, [(A, C)], [], [(B, D)],
              f"smallest angle {m0:.0f}°", f"smallest angle {m1:.0f}°")

    svg.text(W / 2, H - 16, "L: the target edge (the median edge at the start). A collapse keeps the link condition and never leaves a vertex of valence 3; flips stay inside a region.",
             size=12, color=DIM_TEXT, anchor="middle")
    svg.save("remesh-operations.svg")


def draw_pair(svg, x, before, after, hot, dots, new_edges, label0, label1):
    """Before (top) and after (bottom) of one operation, in a panel at x."""
    pts = [v for t in before + after for v in t]
    top = fit(pts, (x + 10, 40, 344, 130))
    bottom = fit(pts, (x + 10, 196, 344, 130))
    for t in before:
        svg.polygon([top(v) for v in t], FACE, stroke=WIRE, width=1.2)
    for a, b in hot:
        svg.line(top(a), top(b), HIGHLIGHT, 3.2)
    svg.text(x + 182, 182, label0, size=12, color=DIM_TEXT, anchor="middle")
    svg.arrow((x + 330, 150), (x + 330, 210), DIM_TEXT, 1.6, 8)
    for t in after:
        svg.polygon([bottom(v) for v in t], FACE, stroke=WIRE, width=1.2)
    for a, b in new_edges:
        svg.line(bottom(a), bottom(b), NEW, 3.0)
    for v in dots:
        svg.circle(bottom(v), 5, VERTEX)
    svg.text(x + 182, 344, label1, size=12, color=DIM_TEXT, anchor="middle")


# ---- A planar test mesh -------------------------------------------------------------------------------


def stretched_grid(nx=15, ny=10, seed=3):
    rng = random.Random(seed)
    P = []
    for j in range(ny):
        for i in range(nx):
            x = i + rng.uniform(-0.22, 0.22) if 0 < i < nx - 1 else i
            y = j + rng.uniform(-0.22, 0.22) if 0 < j < ny - 1 else j
            # stretched 1.6x across the middle columns (and squeezed toward the sides): s' = 1 - 0.6 cos(2 pi x / w)
            s = x - 0.6 * (nx - 1) / (2 * math.pi) * math.sin(2 * math.pi * x / (nx - 1))
            P.append((s, y))
    T = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00, v10, v01, v11 = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            if (i + j) % 2 == 0:
                T += [(v00, v10, v11), (v00, v11, v01)]
            else:
                T += [(v00, v10, v01), (v10, v11, v01)]
    return P, T


def edges_of(T):
    E = {}
    for f, t in enumerate(T):
        for k in range(3):
            e = tuple(sorted((t[k], t[(k + 1) % 3])))
            E.setdefault(e, []).append(f)
    return E


def lloyd_charts(P, T, k, seed, iterations=3):
    cents = [mul(add(add(P[t[0]], P[t[1]]), P[t[2]]), 1 / 3) for t in T]
    rng = random.Random(seed)
    seeds = [cents[i] for i in rng.sample(range(len(T)), k)]
    for _ in range(iterations + 1):
        label = [min(range(k), key=lambda s: norm(sub(c, seeds[s]))) for c in cents]
        for s in range(k):
            members = [cents[f] for f in range(len(T)) if label[f] == s]
            if members:
                seeds[s] = (sum(p[0] for p in members) / len(members), sum(p[1] for p in members) / len(members))
    return label


def chart_borders(T, label):
    out = []
    for e, fs in edges_of(T).items():
        if len(fs) == 2 and label[fs[0]] != label[fs[1]]:
            out.append(e)
    return out


def figure_parallel():
    W, H = 1140, 430
    svg = Svg(W, H)
    P, T = stretched_grid()
    E = edges_of(T)
    L = 1.0
    pw, ph, gap = 364, 374, 14
    titles = ["(a) Independent sets: splits", "(b) Charts, each on its own thread", "(c) Then the band along the borders"]
    for k in range(3):
        svg.panel(14 + k * (pw + gap), 14, pw, ph, titles[k])

    # (a) candidates claim their triangles, longest first; winners hold every claim.
    box = (24, 44, 344, 300)
    f = fit(P, box, 0.04)
    cand = sorted([e for e in E if norm(sub(P[e[0]], P[e[1]])) > 1.5 * L], key=lambda e: -norm(sub(P[e[0]], P[e[1]])))
    claim = {}
    for rank, e in enumerate(cand):
        for face in E[e]:
            claim[face] = min(claim.get(face, rank), rank)
    winners = [e for r, e in enumerate(cand) if all(claim[face] == r for face in E[e])]
    losers = [e for e in cand if e not in winners]
    faces = {}
    for e in winners:
        for face in E[e]:
            faces[tuple(T[face])] = "#24413A"
    draw_mesh(svg, P, T, f, width=0.9, faces=faces)
    for e in losers:
        svg.line(f(P[e[0]]), f(P[e[1]]), LOSER, 2.4, dash="5,4")
    for e in winners:
        svg.line(f(P[e[0]]), f(P[e[1]]), NEW, 3.0)
        svg.circle(f(mid(P[e[0]], P[e[1]])), 3.2, VERTEX)
    svg.text(24 + 6, 372, f"{len(cand)} edges over 1.5 L: {len(winners)} split now (green), {len(losers)} wait", size=12, color=DIM_TEXT)

    # (b) charts; one chart with its frozen ring.
    label = lloyd_charts(P, T, 6, seed=5)
    show = label[len(T) // 2 + 6]
    chart_vertexes = {v for face, t in enumerate(T) if label[face] == show for v in t}
    ring = [face for face, t in enumerate(T) if label[face] != show and any(v in chart_vertexes for v in t)]
    x1 = 14 + pw + gap
    f = fit(P, (x1 + 10, 44, 344, 300), 0.04)
    faces = {tuple(t): CHARTS[label[face] % len(CHARTS)] for face, t in enumerate(T)}
    for face in ring:
        faces[tuple(T[face])] = FROZEN
    for face, t in enumerate(T):
        op = 1.0 if (label[face] == show or face in ring) else 0.45
        svg.polygon([f(P[i]) for i in t], faces[tuple(t)], opacity=op, stroke=PANEL, width=0.8)
    borders_b = chart_borders(T, label)
    for e in borders_b:
        svg.line(f(P[e[0]]), f(P[e[1]]), TEXT, 2.2)
    for e, fs in edges_of(T).items():  # the chart being remeshed, outlined
        inner = [face for face in fs if label[face] == show]
        if len(inner) == 1 and (len(fs) == 1 or label[fs[0]] != label[fs[-1]]):
            svg.line(f(P[e[0]]), f(P[e[1]]), VERTEX, 3.0)
    svg.text(x1 + 16, 372, "every chart on its own thread; grey: the frozen ring of one", size=12, color=DIM_TEXT)

    # (c) the band along the chart borders (triangles with a vertex on a border), the rest frozen.
    border_v = set()
    first = {}
    for face, t in enumerate(T):
        for v in t:
            if v not in first:
                first[v] = label[face]
            elif first[v] != label[face]:
                border_v.add(v)
    band = [face for face, t in enumerate(T) if any(v in border_v for v in t)]
    x2 = 14 + 2 * (pw + gap)
    f = fit(P, (x2 + 10, 44, 344, 300), 0.04)
    for face, t in enumerate(T):
        inband = face in band
        svg.polygon([f(P[i]) for i in t], CHARTS[label[face] % len(CHARTS)] if inband else FROZEN,
                    opacity=1.0 if inband else 0.55, stroke=PANEL, width=0.8)
    for e in borders_b:
        svg.line(f(P[e[0]]), f(P[e[1]]), TEXT, 2.2)
    svg.text(x2 + 16, 372, f"{len(band)} of {len(T)} triangles in the band; grey: frozen", size=12, color=DIM_TEXT)
    svg.text(W / 2, H - 18, "A split claims its two triangles; a collapse the triangles around both ends; a flip its four vertexes. Charts are made once; new triangles join their chart.",
             size=12, color=DIM_TEXT, anchor="middle")
    svg.save("remesh-parallel.svg")


if __name__ == "__main__":
    figure_operations()
    figure_parallel()
