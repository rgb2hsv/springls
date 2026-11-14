#!/usr/bin/env python3
"""Generates docs/springls-badge.svg: the project badge, a constellation of 28 springls on a
circle that the 2D Enright single-vortex flow has started to wind into a swirl.

The shape is computed, not drawn by hand: marker points on the Enright circle (center
(0.5, 0.75), radius 0.15) are advected through the vortex with RK4, then resampled at equal
arc length. Each springl is a short arm along the tangent with a particle in the middle and an
outward normal; the curve itself is never drawn, so the gaps between springls show. Dashed
trails are the particles' paths over the last part of the advection.
Run from anywhere: python3 docs/make_badge.py
"""
import math
from pathlib import Path

N_SPRINGLS = 28
SIZE = 120.0                 # viewBox edge
MARGIN = 9.0                  # room for normals and particles at the edge
T_END = 0.3                  # advection time (Enright period is 8; this is an early swirl)
T_TRAIL = 0.2                # trail length, in advection time
STEPS = 400
ARM_FRACTION = 0.55          # arm length / spacing; < 1 leaves a visible gap
NORMAL_LEN = 6.5
TILT_DEG = -40.0             # turns the tall, narrow shape onto the diagonal to fill the square

PRIMAL, DUAL = "#e57373", "#7fa7d9"
VERTEX, ARM, NORMAL, TRAIL = "#f2c94c", "#9a9a9a", "#7bd88f", "#8b919e"


def velocity(p):
    x, y = p
    return (-math.sin(math.pi * x) ** 2 * math.sin(2 * math.pi * y),
            math.sin(math.pi * y) ** 2 * math.sin(2 * math.pi * x))


def rk4(p, dt):
    def add(a, b, s):
        return (a[0] + s * b[0], a[1] + s * b[1])
    k1 = velocity(p)
    k2 = velocity(add(p, k1, dt / 2))
    k3 = velocity(add(p, k2, dt / 2))
    k4 = velocity(add(p, k3, dt))
    return (p[0] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            p[1] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def advect(p, t, steps):
    if steps == 0:
        return p
    dt = t / steps
    for _ in range(steps):
        p = rk4(p, dt)
    return p


def deformed_curve(markers=600):
    """The Enright circle after T_END, as a dense closed polygon (unit-square coordinates)."""
    pts = [(0.5 + 0.15 * math.cos(2 * math.pi * i / markers),
            0.75 + 0.15 * math.sin(2 * math.pi * i / markers)) for i in range(markers)]
    return [advect(p, T_END, STEPS) for p in pts]


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
    poly = deformed_curve()
    samples, spacing = resample(poly, N_SPRINGLS)
    ccw = signed_area(poly) > 0
    trails = [[advect(p, -T_TRAIL * k / 12, 8 * k) for k in range(13)] for p, _ in samples]
    # Tilt the long axis onto the diagonal, then fit curve and trails into the viewBox
    # (y flipped: SVG y grows downward).
    c, s = math.cos(math.radians(TILT_DEG)), math.sin(math.radians(TILT_DEG))
    rot = lambda p: (c * p[0] - s * p[1], s * p[0] + c * p[1])
    fit = [rot(q) for q in poly + [q for t in trails for q in t]]
    xs, ys = [p[0] for p in fit], [p[1] for p in fit]
    scale = (SIZE - 2 * MARGIN) / max(max(xs) - min(xs), max(ys) - min(ys))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def to_svg(p):
        x, y = rot(p)
        return (SIZE / 2 + (x - cx) * scale, SIZE / 2 - (y - cy) * scale)

    springls = []
    for (p, (tx, ty)), trail in zip(samples, trails):
        n = (ty, -tx) if ccw else (-ty, tx)          # outward normal in unit-square coords
        h = ARM_FRACTION * spacing / 2
        a, b = (p[0] - h * tx, p[1] - h * ty), (p[0] + h * tx, p[1] + h * ty)
        springls.append((p, a, b, n, trail))
    return poly, springls, spacing, to_svg, scale


def render(springls, to_svg, scale):
    f = lambda v: f"{v:.2f}"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE:.0f} {SIZE:.0f}" '
           f'width="{SIZE:.0f}" height="{SIZE:.0f}" role="img" aria-label="Springls badge">',
           '<title>Springls: a deforming surface carried by a constellation of springls</title>']
    out.append(f'<g fill="none" stroke="{TRAIL}" stroke-width="0.8" stroke-dasharray="1.6 2.2" '
               f'stroke-linecap="round" opacity="0.45">')
    for *_, trail in springls:
        out.append('<polyline points="' + " ".join(f"{f(x)},{f(y)}" for x, y in map(to_svg, trail)) + '"/>')
    out.append('</g>')
    for i, (p, a, b, n, _) in enumerate(springls):
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
    poly, springls, spacing, to_svg, scale = build()
    svg = render(springls, to_svg, scale)
    # Self-checks: count, visible gaps, outward normals, nothing GitHub would refuse to render.
    assert len(springls) == N_SPRINGLS
    for p, a, b, n, _ in springls:
        assert math.dist(a, b) < 0.8 * spacing, "arms too long: gaps would close"
        probe = (p[0] + 0.25 * spacing * n[0], p[1] + 0.25 * spacing * n[1])
        assert not inside(probe, poly), "normal points inward"
    assert all(s not in svg for s in ("href", "<script", "<image"))
    path = Path(__file__).resolve().parent / "springls-badge.svg"
    path.write_text(svg)
    print("wrote docs/springls-badge.svg")


if __name__ == "__main__":
    main()
