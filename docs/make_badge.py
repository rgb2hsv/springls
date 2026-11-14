#!/usr/bin/env python3
"""Generates docs/springls-badge.svg: the project badge, a constellation of 28 springls on a
deformed circle, with a dashed outline of the circle it started as.

The deformed shape is the blob from the "Semi-Lagrangian doubleweave: one sub-step" figure
(make_semi_lagrangian_diagram.py): r(theta) = 12 + 2.2 cos(3 theta + 0.5). The springls sit at
equal arc length around it; each is a short arm along the tangent with a particle in the middle
and an outward normal. The curve itself is never drawn, so the gaps between springls show. The
dashed circle (radius 12, offset down and to the left) is where the surface was before it moved
and deformed. Run from anywhere: python3 docs/make_badge.py
"""
import math
from pathlib import Path

N_SPRINGLS = 28
SIZE = 120.0                 # viewBox edge
MARGIN = 9.0                 # room for normals and particles at the edge
R0, LOBE, PHASE = 12.0, 2.2, 0.5   # the one-sub-step figure's blob
BEFORE_OFFSET = (-4.5, -3.5)       # the undeformed circle, relative to the blob (y up)
ARM_FRACTION = 0.55          # arm length / spacing; < 1 leaves a visible gap
NORMAL_LEN = 6.5

PRIMAL, DUAL = "#e57373", "#7fa7d9"
VERTEX, ARM, NORMAL, BEFORE = "#f2c94c", "#9a9a9a", "#7bd88f", "#8b919e"


def blob(theta):
    r = R0 + LOBE * math.cos(3 * theta + PHASE)
    return (r * math.cos(theta), r * math.sin(theta))


def polygon(f, markers=720):
    return [f(2 * math.pi * i / markers) for i in range(markers)]


def resample(poly, n):
    """n points at equal arc length around a closed polygon, with unit tangents."""
    seg = [math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    total, out, i, acc = sum(seg), [], 0, 0.0
    for k in range(n):
        s = k * total / n
        while acc + seg[i] < s:
            acc += seg[i]
            i += 1
        a, b = poly[i], poly[(i + 1) % len(poly)]
        f = (s - acc) / seg[i]
        tx, ty = (b[0] - a[0]) / seg[i], (b[1] - a[1]) / seg[i]
        out.append(((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), (tx, ty)))
    return out, total / n


def signed_area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] -
                     poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def inside(q, poly):
    c = False
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if (a[1] > q[1]) != (b[1] > q[1]) and q[0] < a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            c = not c
    return c


def build():
    poly = polygon(blob)
    before = polygon(lambda t: (BEFORE_OFFSET[0] + R0 * math.cos(t), BEFORE_OFFSET[1] + R0 * math.sin(t)), 180)
    samples, spacing = resample(poly, N_SPRINGLS)
    ccw = signed_area(poly) > 0
    # Fit the blob and the dashed circle into the viewBox (y flipped: SVG y grows downward).
    xs, ys = [p[0] for p in poly + before], [p[1] for p in poly + before]
    scale = (SIZE - 2 * MARGIN - NORMAL_LEN) / max(max(xs) - min(xs), max(ys) - min(ys))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def to_svg(p):
        return (SIZE / 2 + (p[0] - cx) * scale, SIZE / 2 - (p[1] - cy) * scale)

    springls = []
    for p, (tx, ty) in samples:
        n = (ty, -tx) if ccw else (-ty, tx)          # outward normal
        h = ARM_FRACTION * spacing / 2
        springls.append((p, (p[0] - h * tx, p[1] - h * ty), (p[0] + h * tx, p[1] + h * ty), n))
    return poly, before, springls, spacing, to_svg, scale


def render(before, springls, to_svg, scale):
    f = lambda v: f"{v:.2f}"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE:.0f} {SIZE:.0f}" '
           f'width="{SIZE:.0f}" height="{SIZE:.0f}" role="img" aria-label="Springls badge">',
           '<title>Springls: a deforming surface carried by a constellation of springls</title>',
           f'<polygon points="' + " ".join(f"{f(x)},{f(y)}" for x, y in map(to_svg, before)) +
           f'" fill="none" stroke="{BEFORE}" stroke-width="1" stroke-dasharray="2.5 2.5" opacity="0.6"/>']
    for i, (p, a, b, n) in enumerate(springls):
        P, A, B = to_svg(p), to_svg(a), to_svg(b)
        tip = to_svg((p[0] + n[0] * NORMAL_LEN / scale, p[1] + n[1] * NORMAL_LEN / scale))
        out.append(f'<line x1="{f(P[0])}" y1="{f(P[1])}" x2="{f(tip[0])}" y2="{f(tip[1])}" '
                   f'stroke="{NORMAL}" stroke-width="1.1" stroke-linecap="round"/>')
        out.append(f'<line x1="{f(A[0])}" y1="{f(A[1])}" x2="{f(B[0])}" y2="{f(B[1])}" '
                   f'stroke="{ARM}" stroke-width="1.7" stroke-linecap="round"/>')
        for V in (A, B):
            out.append(f'<circle cx="{f(V[0])}" cy="{f(V[1])}" r="1.3" fill="{VERTEX}"/>')
        out.append(f'<circle cx="{f(P[0])}" cy="{f(P[1])}" r="2.3" fill="{PRIMAL if i % 2 == 0 else DUAL}"/>')
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main():
    poly, before, springls, spacing, to_svg, scale = build()
    svg = render(before, springls, to_svg, scale)
    # Self-checks: count, visible gaps, outward normals, nothing GitHub would refuse to render.
    assert len(springls) == N_SPRINGLS
    for p, a, b, n in springls:
        assert math.dist(a, b) < 0.8 * spacing, "arms too long: gaps would close"
        probe = (p[0] + 0.25 * spacing * n[0], p[1] + 0.25 * spacing * n[1])
        assert not inside(probe, poly), "normal points inward"
    assert all(s not in svg for s in ("href", "<script", "<image"))
    path = Path(__file__).resolve().parent / "springls-badge.svg"
    path.write_text(svg)
    print("wrote docs/springls-badge.svg")


if __name__ == "__main__":
    main()
