#!/usr/bin/env python3
"""Generates the intrinsic springls figures of the documentation (section 3.14):

- docs/intrinsic-flip.svg: an edge flip on an intrinsic triangulation. A folded hinge of the input mesh
  whose shared edge is not Delaunay, the hinge laid out in the plane from its four edge lengths alone,
  and the flipped edge drawn back on the unchanged surface, where it bends over the old edge.
- docs/intrinsic-trace.svg: the discrete exponential map. A strip of folded faces, a straight line
  traced across it with a tangent vector carried along (parallel transport), and the same strip
  unfolded, where the path is a straight line and the vector is constant.
- docs/intrinsic-reconstruction.svg: one semi-Lagrangian rebuild on a triangle mesh. A puzzle piece's
  springls (one missing), the mesh refined around them (Loop 1:4 splits closed red-green), the
  unsigned distance U per vertex with the walls (U <= capsule), the sign by the exterior flood and the
  upwind sweep, and phi with its marching-triangles contour, which bridges the gap.

The panels are computed, not drawn by hand: small re-implementations of the flip, the trace, the
refinement and the sign rule (IntrinsicTriangulation, IntrinsicTrace, IntrinsicSignField). The
reconstruction runs on a flat patch, where geodesic and Euclidean distance agree, so U is exact.
Colors are the toybox palette (apps/windows/common/Palette.hpp).
Run from anywhere: python3 docs/make_intrinsic_diagrams.py
"""
import heapq
import math
from collections import deque
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Palette (apps/windows/common/Palette.hpp).
BACKGROUND = "#1E1F24"
PANEL = "#15161A"
TEXT = "#E6E6E6"
DIM_TEXT = "#A8A8B0"
CONTOUR = "#3FC1C9"
SPRINGL = "#8A93A6"
PARTICLE = "#D23A48"
PARTICLE_LABEL = "#FF8A94"  # PARTICLE lightened for text on the dark panels (contrast about 7:1)
VERTEX = "#F5C542"
ARROW = "#9C1B74"
ARROW_LIGHT = "#E048B8"
REFERENCE = "#F2EBD3"
OBJECT = "#E8E4DA"
LEVELS = ["#5B6073", "#3E6F94", "#5560AA", "#7D5AA6", "#A65A9A"]
# Sign classes.
FLOOD_OUT, UNREACHED_IN = "#26272E", "#1F5E62"
WALL_OUT, WALL_IN = "#5C5C70", "#2F9C96"
FALLBACK_IN = "#8A5CC8"
POCKET_OUT = "#4A3F66"
CORRECTED = "#E3B65C"
FONT = "Inter, Helvetica, Arial, sans-serif"


# ---- small vector helpers ---------------------------------------------------------------------------
def sub(a, b): return tuple(x - y for x, y in zip(a, b))
def add(a, b): return tuple(x + y for x, y in zip(a, b))
def mul(a, s): return tuple(x * s for x in a)
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def unit(a): return mul(a, 1.0 / norm(a))
def lerp(a, b, t): return add(a, mul(sub(b, a), t))
def cross3(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def cross2(a, b): return a[0] * b[1] - a[1] * b[0]


def rotate_about_axis(p, origin, axis, angle):
    """Rodrigues rotation of point p about the line through origin along unit axis."""
    v = sub(p, origin)
    c, s = math.cos(angle), math.sin(angle)
    r = add(add(mul(v, c), mul(cross3(axis, v), s)), mul(axis, dot(axis, v) * (1 - c)))
    return add(origin, r)


class Svg:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.parts = [f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>']

    def add(self, s):
        self.parts.append(s)

    def panel(self, x, y, w, h, title):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PANEL}"/>')
        self.text(x + 14, y + 26, title, size=15, weight=600)

    def text(self, x, y, s, size=13, color=TEXT, anchor="start", weight=400, italic=False, halo=False):
        style = ' font-style="italic"' if italic else ""
        if halo:  # a dark outline under the glyphs, for labels drawn over a busy fill
            style += f' stroke="{PANEL}" stroke-width="3.5" stroke-linejoin="round" paint-order="stroke"'
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{style}>{s}</text>')

    def line(self, a, b, color, width=1.5, dash=None, opacity=1.0, cap="round"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{color}" '
                 f'stroke-width="{width}" stroke-linecap="{cap}" opacity="{opacity}"{d}/>')

    def polyline(self, pts, color, width=1.5, dash=None, opacity=1.0, closed=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        tag = "polygon" if closed else "polyline"
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<{tag} points="{p}" fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{d}/>')

    def polygon(self, pts, fill, opacity=1.0, stroke="none", width=0.0):
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<polygon points="{p}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" '
                 f'stroke-width="{width}" stroke-linejoin="round"/>')

    def circle(self, c, r, fill, stroke="none", width=0.0):
        self.add(f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def arrow(self, a, b, color, width=2.0, head=9.0):
        d = sub(b, a)
        if norm(d) < 1e-9:
            return
        u = unit(d)
        n = (-u[1], u[0])
        base = sub(b, mul(u, head))
        self.line(a, base, color, width)
        self.polygon([b, add(base, mul(n, head * 0.42)), sub(base, mul(n, head * 0.42))], color)

    def arc(self, center, radius, a0, a1, color, width=1.4):
        """Arc in screen coordinates from angle a0 to a1 (radians, screen y down)."""
        p0 = (center[0] + radius * math.cos(a0), center[1] + radius * math.sin(a0))
        p1 = (center[0] + radius * math.cos(a1), center[1] + radius * math.sin(a1))
        large = 1 if abs(a1 - a0) > math.pi else 0
        sweep = 1 if a1 > a0 else 0
        self.add(f'<path d="M{p0[0]:.2f},{p0[1]:.2f} A{radius},{radius} 0 {large} {sweep} {p1[0]:.2f},{p1[1]:.2f}" '
                 f'fill="none" stroke="{color}" stroke-width="{width}"/>')

    def save(self, name):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
               f'viewBox="0 0 {self.width} {self.height}">\n{body}\n</svg>\n')
        (OUT / name).write_text(svg)
        print(f"wrote {OUT / name}")


class Camera:
    """Orthographic view of 3D points, fitted into a screen box."""

    def __init__(self, yaw, pitch):
        self.yaw, self.pitch = yaw, pitch
        self.scale, self.offset = 1.0, (0.0, 0.0)

    def raw(self, p):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x, y, z = p
        x, y = cy * x - sy * y, sy * x + cy * y
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        y, z = cp * y - sp * z, sp * y + cp * z
        return (x, -z, y)  # screen x, screen y (down), depth

    def fit(self, points, box, margin=0.1):
        pts = [self.raw(p) for p in points]
        x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
        y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
        bx, by, bw, bh = box
        self.scale = min(bw * (1 - 2 * margin) / (x1 - x0), bh * (1 - 2 * margin) / (y1 - y0))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.offset = (bx + bw / 2 - cx * self.scale, by + bh / 2 - cy * self.scale)

    def __call__(self, p):
        r = self.raw(p)
        return (self.offset[0] + r[0] * self.scale, self.offset[1] + r[1] * self.scale)

    def depth(self, p):
        return self.raw(p)[2]


def fit2d(points, box, margin=0.1, flip_y=True):
    x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
    y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
    bx, by, bw, bh = box
    s = min(bw * (1 - 2 * margin) / (x1 - x0), bh * (1 - 2 * margin) / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sy = -1 if flip_y else 1
    return lambda p: (bx + bw / 2 + (p[0] - cx) * s, by + bh / 2 + sy * (p[1] - cy) * s)


def layout_third(p0, p1, l0, l1, side=1):
    """2D point at distance l0 from p0 and l1 from p1, on the left (side 1) or right (-1) of p0 -> p1."""
    d = norm(sub(p1, p0))
    x = (l0 * l0 - l1 * l1 + d * d) / (2 * d)
    y = math.sqrt(max(l0 * l0 - x * x, 0.0))
    u = unit(sub(p1, p0))
    n = (-u[1], u[0])
    return add(add(p0, mul(u, x)), mul(n, side * y))


def angle_at(p, a, b):
    u, v = unit(sub(a, p)), unit(sub(b, p))
    return math.acos(max(-1.0, min(1.0, dot(u, v))))


def shade(color, normal3, light=(0.35, -0.45, 0.82)):
    """Lambert-ish shading of a hex color by a 3D normal."""
    lam = abs(dot(unit(normal3), unit(light)))
    f = 0.55 + 0.45 * lam
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return "#%02X%02X%02X" % (int(r * f), int(g * f), int(b * f))


# =====================================================================================================
# Figure: edge flip
# =====================================================================================================
def figure_flip():
    # A hinge of the input mesh: faces i j k and j i l share edge ij, folded along it. k and l lie close
    # to ij, so the two angles opposite ij sum to more than π: ij is not Delaunay.
    i3, j3 = (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)
    k_flat, l_flat = (1.6, 1.05, 0.0), (2.5, -0.95, 0.0)
    axis = unit(sub(j3, i3))
    k3 = rotate_about_axis(k_flat, i3, axis, -0.6)
    l3 = rotate_about_axis(l_flat, i3, axis, -0.75)

    # Lengths are all an intrinsic triangulation keeps.
    lij, lik, ljk = norm(sub(j3, i3)), norm(sub(k3, i3)), norm(sub(k3, j3))
    lil, ljl = norm(sub(l3, i3)), norm(sub(l3, j3))
    # Layout from lengths: i at the origin, j on +x, k above, l below.
    i2, j2 = (0.0, 0.0), (lij, 0.0)
    k2 = layout_third(i2, j2, lik, ljk, 1)
    l2 = layout_third(i2, j2, lil, ljl, -1)
    alpha, beta = angle_at(k2, i2, j2), angle_at(l2, i2, j2)
    lkl = norm(sub(l2, k2))
    # Where the new edge crosses ij, as a parameter along ij (the same on the surface).
    d = sub(l2, k2)
    t_cross = (k2[1] * 0 - k2[1]) / d[1]  # k2.y + t d.y = 0
    x_cross = k2[0] + t_cross * d[0]
    s_cross = x_cross / lij
    cross3d = lerp(i3, j3, s_cross)

    W, H = 1140, 420
    svg = Svg(W, H)
    pw, ph, gap = 360, 380, 15
    boxes = [(gap + n * (pw + gap), 20, pw, ph) for n in range(3)]

    cam = Camera(yaw=0.3, pitch=-0.45)
    cam.fit([i3, j3, k3, l3], (boxes[0][0], boxes[0][1] + 40, pw, ph - 110), margin=0.12)

    def draw_hinge(box, cam, flipped):
        faces = [(i3, j3, k3), (j3, i3, l3)]
        for f in faces:
            n3 = cross3(sub(f[1], f[0]), sub(f[2], f[0]))
            svg.polygon([cam(p) for p in f], shade("#6E7488", n3), opacity=0.9)
        edges = [(i3, k3), (j3, k3), (i3, l3), (j3, l3)]
        for a, b in edges:
            svg.line(cam(a), cam(b), SPRINGL, 1.6)
        if flipped:
            svg.line(cam(i3), cam(j3), SPRINGL, 1.2, dash="4 4", opacity=0.8)
            svg.polyline([cam(k3), cam(cross3d), cam(l3)], CONTOUR, 2.6)
            svg.circle(cam(cross3d), 3.2, CONTOUR)
        else:
            svg.line(cam(i3), cam(j3), PARTICLE, 2.6)
        for p, name, off in [(i3, "i", (-14, 4)), (j3, "j", (8, 4)), (k3, "k", (-4, -10)), (l3, "l", (-4, 20))]:
            q = cam(p)
            svg.circle(q, 4, VERTEX)
            svg.text(q[0] + off[0], q[1] + off[1], name, 15, TEXT, italic=True)

    # (a) the input hinge.
    bx, by, _, _ = boxes[0]
    svg.panel(*boxes[0], "(a) Input mesh: a folded hinge")
    draw_hinge(boxes[0], cam, flipped=False)
    svg.text(bx + 14, by + ph - 52, f"α + β = {math.degrees(alpha + beta):.0f}° &gt; 180°: edge ij is", 13, DIM_TEXT)
    svg.text(bx + 14, by + ph - 32, "not Delaunay (cotan weight &lt; 0)", 13, DIM_TEXT)

    # (b) layout from lengths.
    bx, by, _, _ = boxes[1]
    svg.panel(*boxes[1], "(b) Laid out from edge lengths")
    to = fit2d([i2, j2, k2, l2], (bx, by + 40, pw, ph - 110), margin=0.14)
    svg.polygon([to(i2), to(j2), to(k2)], "#5B6073", opacity=0.55)
    svg.polygon([to(j2), to(i2), to(l2)], "#4E5466", opacity=0.55)
    for a, b in [(i2, k2), (j2, k2), (i2, l2), (j2, l2)]:
        svg.line(to(a), to(b), SPRINGL, 1.6)
    svg.line(to(i2), to(j2), PARTICLE, 1.6, dash="5 4")
    svg.line(to(k2), to(l2), CONTOUR, 2.6)
    # Opposite angles.
    for p, a, b, name in [(k2, i2, j2, "α"), (l2, j2, i2, "β")]:
        q = to(p)
        qa, qb = to(a), to(b)
        a0 = math.atan2(qa[1] - q[1], qa[0] - q[0])
        a1 = math.atan2(qb[1] - q[1], qb[0] - q[0])
        if a1 < a0:
            a0, a1 = a1, a0
        if a1 - a0 > math.pi:
            a0, a1 = a1, a0 + 2 * math.pi
        svg.arc(q, 20, a0, a1, VERTEX, 1.4)
        mid = (a0 + a1) / 2
        svg.text(q[0] + 34 * math.cos(mid), q[1] + 34 * math.sin(mid) + 5, name, 14, VERTEX, "middle", italic=True)
    for p, name, off in [(i2, "i", (-16, 5)), (j2, "j", (9, 5)), (k2, "k", (-4, -10)), (l2, "l", (-4, 22))]:
        q = to(p)
        svg.circle(q, 4, VERTEX)
        svg.text(q[0] + off[0], q[1] + off[1], name, 15, TEXT, italic=True)
    svg.text(bx + 14, by + ph - 52, f"flip: new edge kl, length |k − l| = {lkl:.2f}", 13, DIM_TEXT)
    svg.text(bx + 14, by + ph - 32, "in this layout (no 3D positions used)", 13, DIM_TEXT)

    # (c) the flipped edge on the same surface.
    bx, by, _, _ = boxes[2]
    svg.panel(*boxes[2], "(c) Intrinsic edge on the same surface")
    cam2 = Camera(yaw=0.3, pitch=-0.45)
    cam2.fit([i3, j3, k3, l3], (bx, by + 40, pw, ph - 110), margin=0.12)
    draw_hinge(boxes[2], cam2, flipped=True)
    svg.text(bx + 14, by + ph - 52, "kl is a geodesic: straight in each face,", 13, DIM_TEXT)
    svg.text(bx + 14, by + ph - 32, "bending where it crosses the old edge ij", 13, DIM_TEXT)
    svg.save("intrinsic-flip.svg")


# =====================================================================================================
# Figure: discrete exponential map
# =====================================================================================================
def figure_trace():
    # A strip of faces laid out in the plane (the unfolding), then folded along every shared edge.
    flat = [(0.0, 0.0), (0.4, 2.0), (2.0, -0.3), (2.6, 2.2), (4.1, 0.1), (4.7, 2.1), (6.3, -0.2), (6.8, 2.0), (8.4, 0.2)]
    tris = [(n, n + 1, n + 2) if n % 2 == 0 else (n + 1, n, n + 2) for n in range(len(flat) - 2)]
    folds = [0.0, 0.42, -0.5, 0.45, -0.48, 0.4, -0.45]  # fold of face n relative to face n - 1
    # 3D positions: fold successive faces about their shared edge.
    pos3 = [(x, y, 0.0) for x, y in flat]
    for n in range(1, len(tris)):
        a, b = pos3[n], pos3[n + 1]  # shared edge of faces n - 1 and n: vertexes n and n + 1
        ax = unit(sub(b, a))
        for m in range(n + 2, len(flat)):
            pos3[m] = rotate_about_axis(pos3[m], a, ax, folds[n])

    def to3d(face, p2):
        """Map a point of the flat strip in face `face` to the folded surface (barycentric)."""
        a, b, c = tris[face]
        pa, pb, pc = flat[a], flat[b], flat[c]
        den = cross2(sub(pb, pa), sub(pc, pa))
        wb = cross2(sub(p2, pa), sub(pc, pa)) / den
        wc = cross2(sub(pb, pa), sub(p2, pa)) / den
        wa = 1 - wb - wc
        return add(add(mul(pos3[a], wa), mul(pos3[b], wb)), mul(pos3[c], wc))

    def face_of(p2):
        for f, (a, b, c) in enumerate(tris):
            pa, pb, pc = flat[a], flat[b], flat[c]
            s1, s2, s3 = cross2(sub(pb, pa), sub(p2, pa)), cross2(sub(pc, pb), sub(p2, pb)), cross2(sub(pa, pc), sub(p2, pc))
            if (s1 >= -1e-9 and s2 >= -1e-9 and s3 >= -1e-9) or (s1 <= 1e-9 and s2 <= 1e-9 and s3 <= 1e-9):
                return f
        return None

    start, disp = (0.55, 0.65), (6.6, 0.6)
    end = add(start, disp)
    # Sample the straight line and its crossings with the faces.
    samples = []
    for s in range(0, 401):
        p = add(start, mul(disp, s / 400))
        f = face_of(p)
        if f is not None:
            samples.append((f, p))
    vec = (0.0, 0.85)  # tangent vector at the start, carried along (constant in the unfolding)

    W, H = 1140, 470
    svg = Svg(W, H)
    pw, ph, gap = 555, 430, 10
    box_a, box_b = (gap, 20, pw, ph), (2 * gap + pw, 20, pw, ph)

    # (a) on the folded surface.
    svg.panel(*box_a, "(a) Trace on the surface: Trace(p, v)")
    cam = Camera(yaw=-0.15, pitch=-0.75)
    cam.fit(pos3, (box_a[0], box_a[1] + 40, pw, ph - 110), margin=0.06)
    order = sorted(range(len(tris)), key=lambda f: -sum(cam.depth(pos3[v]) for v in tris[f]))
    for f in order:
        a, b, c = tris[f]
        n3 = cross3(sub(pos3[b], pos3[a]), sub(pos3[c], pos3[a]))
        svg.polygon([cam(pos3[a]), cam(pos3[b]), cam(pos3[c])], shade("#6E7488", n3), opacity=0.95, stroke=SPRINGL, width=1.2)
    pts3 = [to3d(f, p) for f, p in samples]
    svg.polyline([cam(p) for p in pts3], CONTOUR, 2.6)
    # Transported vector at a few places along the path: in each face, the flat vector mapped by that
    # face's (affine) embedding.
    for s in [0.0, 0.33, 0.66, 1.0]:
        p = add(start, mul(disp, s))
        f = face_of(p) if s < 1 else face_of(sub(p, mul(disp, 1e-6)))
        q0, q1 = to3d(f, p), to3d(f, add(p, mul(vec, 0.7)))
        svg.arrow(cam(q0), cam(q1), ARROW_LIGHT, 2.2, 9)
    svg.circle(cam(pts3[0]), 4.5, VERTEX)
    svg.circle(cam(pts3[-1]), 4.5, PARTICLE)
    svg.text(cam(pts3[0])[0] - 12, cam(pts3[0])[1] + 24, "p", 15, TEXT, italic=True)
    svg.text(cam(pts3[-1])[0] + 4, cam(pts3[-1])[1] + 24, "exp<tspan baseline-shift='sub' font-size='10'>p</tspan>(v)", 15, TEXT, italic=True)
    svg.text(box_a[0] + 14, box_a[1] + ph - 52, "Straight in each face; at an edge it continues into the", 13, DIM_TEXT)
    svg.text(box_a[0] + 14, box_a[1] + ph - 32, "next face at the same angle. Vectors ride along unrotated.", 13, DIM_TEXT)

    # (b) unfolded.
    svg.panel(*box_b, "(b) The same faces unfolded")
    to = fit2d(flat, (box_b[0], box_b[1] + 40, pw, ph - 110), margin=0.06)
    for a, b, c in tris:
        svg.polygon([to(flat[a]), to(flat[b]), to(flat[c])], "#4E5466", opacity=0.6, stroke=SPRINGL, width=1.2)
    svg.line(to(start), to(end), CONTOUR, 2.6)
    for s in [0.0, 0.33, 0.66, 1.0]:
        p = add(start, mul(disp, s))
        svg.arrow(to(p), to(add(p, mul(vec, 0.55))), ARROW_LIGHT, 2.2, 9)
    svg.arrow(to(start), to(add(start, mul(disp, 0.16))), VERTEX, 2.2, 10)
    svg.text(to(add(start, mul(disp, 0.08)))[0] - 4, to(start)[1] + 22, "v", 15, VERTEX, italic=True)
    svg.circle(to(start), 4.5, VERTEX)
    svg.circle(to(end), 4.5, PARTICLE)
    # Edge crossings.
    last = samples[0][0]
    for f, p in samples:
        if f != last:
            svg.circle(to(p), 3.0, TEXT)
            last = f
    svg.text(box_b[0] + 14, box_b[1] + ph - 52, "Unfolded, the path is one straight line and the vector is", 13, DIM_TEXT)
    svg.text(box_b[0] + 14, box_b[1] + ph - 32, "constant; dots mark the edge crossings the trace walks.", 13, DIM_TEXT)
    svg.save("intrinsic-trace.svg")


# =====================================================================================================
# Figure: semi-Lagrangian rebuild on a refined triangle mesh
# =====================================================================================================
R = 1.0  # puzzle piece half-side


def puzzle_inside(p):
    """Puzzle piece of PuzzlePieceOutline: square of side 2r, tabs on +x and +y, sockets on -x and -y."""
    x, y = p
    tab_r, off = 0.35 * R, 0.25 * R
    in_sq = abs(x) <= R and abs(y) <= R
    in_socket = math.hypot(x + R - off, y) < tab_r or math.hypot(x, y + R - off) < tab_r
    in_tab = math.hypot(x - R - off, y) < tab_r or math.hypot(x, y - R - off) < tab_r
    return (in_sq and not in_socket) or in_tab


def puzzle_outline(spacing=0.01 * R):
    """The outline of puzzle_inside exactly, as straight sides and circular arcs, counter-clockwise from
    the bottom-left corner, sampled about `spacing` apart."""
    tab_r, off = 0.35 * R, 0.25 * R
    half = math.sqrt(tab_r * tab_r - off * off)  # half the neck's width, where a tab or socket meets its side
    pts = []

    def line(a, b):
        n = max(1, int(math.ceil(norm(sub(b, a)) / spacing)))
        pts.extend(lerp(a, b, k / n) for k in range(n))

    def arc(center, a, b, through):
        """Arc of the circle about center from a to b, the way that passes through `through`."""
        a0 = math.atan2(a[1] - center[1], a[0] - center[0])
        a1 = math.atan2(b[1] - center[1], b[0] - center[0])
        am = math.atan2(through[1] - center[1], through[0] - center[0])
        ccw = lambda x: x % (2 * math.pi)
        span = ccw(a1 - a0)
        if ccw(am - a0) > span:  # `through` is not on the counter-clockwise arc: go clockwise
            span -= 2 * math.pi
        n = max(2, int(math.ceil(abs(span) * tab_r / spacing)))
        pts.extend((center[0] + tab_r * math.cos(a0 + span * k / n), center[1] + tab_r * math.sin(a0 + span * k / n)) for k in range(n))

    # Bottom side, socket cut upward.
    c = (0.0, -R + off)
    line((-R, -R), (-half, -R)); arc(c, (-half, -R), (half, -R), add(c, (0, tab_r))); line((half, -R), (R, -R))
    # Right side, tab outward.
    c = (R + off, 0.0)
    line((R, -R), (R, -half)); arc(c, (R, -half), (R, half), add(c, (tab_r, 0))); line((R, half), (R, R))
    # Top side, tab outward.
    c = (0.0, R + off)
    line((R, R), (half, R)); arc(c, (half, R), (-half, R), add(c, (0, tab_r))); line((-half, R), (-R, R))
    # Left side, socket cut inward.
    c = (-R + off, 0.0)
    line((-R, R), (-R, half)); arc(c, (-R, half), (-R, -half), add(c, (tab_r, 0))); line((-R, -half), (-R, -R))
    return pts


def resample(poly, spacing):
    """Points evenly spaced along a closed polyline, about `spacing` apart."""
    lengths = [norm(sub(poly[(k + 1) % len(poly)], poly[k])) for k in range(len(poly))]
    total = sum(lengths)
    n = max(3, int(round(total / spacing)))
    step = total / n
    out, k, acc = [], 0, 0.0
    for m in range(n):
        target = m * step
        while acc + lengths[k] < target:
            acc += lengths[k]
            k += 1
        t = (target - acc) / lengths[k] if lengths[k] > 0 else 0.0
        out.append(lerp(poly[k], poly[(k + 1) % len(poly)], t))
    return out


def segment_distance(p, a, b):
    ab = sub(b, a)
    t = max(0.0, min(1.0, dot(sub(p, a), ab) / max(dot(ab, ab), 1e-30)))
    return norm(sub(p, add(a, mul(ab, t))))


def point_polyline_distance(p, poly):
    return min(segment_distance(p, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly)))


class Mesh:
    """A flat triangle mesh refined by Loop 1:4 splits (no smoothing) closed red-green."""

    def __init__(self, nx, ny, x0, y0, size):
        self.v = []
        self.level = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                # A staggered lattice (equilateral-ish faces), as the toybox's icosphere looks locally.
                self.v.append((x0 + (i + 0.5 * (j % 2)) * size, y0 + j * size * math.sqrt(3) / 2))
        self.f = []
        for j in range(ny):
            for i in range(nx):
                a, b = j * (nx + 1) + i, j * (nx + 1) + i + 1
                c, d = (j + 1) * (nx + 1) + i, (j + 1) * (nx + 1) + i + 1
                if j % 2 == 0:
                    self.f += [(a, b, c), (b, d, c)]
                else:
                    self.f += [(a, b, d), (a, d, c)]
        self.level = [0] * len(self.f)
        self.mid = {}

    def midpoint(self, a, b):
        key = (min(a, b), max(a, b))
        if key not in self.mid:
            self.v.append(lerp(self.v[a], self.v[b], 0.5))
            self.mid[key] = len(self.v) - 1
        return self.mid[key]

    def has_mid(self, a, b):
        return (min(a, b), max(a, b)) in self.mid

    def refine(self, marks):
        """Splits the marked faces 1:4, then closes: faces with two or more split edges, or an edge whose
        halves are split again (a 2:1 jump), split too."""
        marks = list(marks)
        while True:
            for f, m in enumerate(marks):
                if m:
                    a, b, c = self.f[f]
                    for p, q in [(a, b), (b, c), (c, a)]:
                        self.midpoint(p, q)
            changed = False
            for f, (a, b, c) in enumerate(self.f):
                if marks[f]:
                    continue
                edges = [(a, b), (b, c), (c, a)]
                split = sum(self.has_mid(p, q) for p, q in edges)
                deep = False
                for p, q in edges:
                    if self.has_mid(p, q):
                        m = self.mid[(min(p, q), max(p, q))]
                        if self.has_mid(p, m) or self.has_mid(m, q):
                            deep = True
                if split >= 2 or deep:
                    marks[f] = 1
                    changed = True
            if not changed:
                break
        faces, levels = [], []
        for f, (a, b, c) in enumerate(self.f):
            if marks[f]:
                ab, bc, ca = self.mid[(min(a, b), max(a, b))], self.mid[(min(b, c), max(b, c))], self.mid[(min(c, a), max(c, a))]
                faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
                levels += [self.level[f] + 1] * 4
            else:
                faces.append((a, b, c))
                levels.append(self.level[f])
        self.f, self.level = faces, levels

    def conforming(self):
        """The faces with the green bisections of faces with one hanging edge midpoint."""
        out = []
        for f, (a, b, c) in enumerate(self.f):
            done = False
            for p, q, r in [(a, b, c), (b, c, a), (c, a, b)]:
                if self.has_mid(p, q):
                    m = self.mid[(min(p, q), max(p, q))]
                    out += [((p, m, r), self.level[f], True), ((m, q, r), self.level[f], True)]
                    done = True
                    break
            if not done:
                out.append(((a, b, c), self.level[f], False))
        return out


def figure_reconstruction():
    h0 = 0.5 * R
    mesh = Mesh(10, 12, -2.55 * R, -2.45 * R, h0)
    outline = puzzle_outline()
    levels = 2
    h_t = h0 / 2 ** levels
    capsule = 1.5 * h_t
    band = capsule + 2.0 * h_t  # kCapsule + kRefineMargin
    for _ in range(levels):
        marks = []
        for a, b, c in mesh.f:
            cen = mul(add(add(mesh.v[a], mesh.v[b]), mesh.v[c]), 1 / 3)
            longest = max(norm(sub(mesh.v[a], mesh.v[b])), norm(sub(mesh.v[b], mesh.v[c])), norm(sub(mesh.v[c], mesh.v[a])))
            marks.append(1 if point_polyline_distance(cen, outline) < band + 0.4 * longest else 0)
        mesh.refine(marks)
    faces = mesh.conforming()
    V = mesh.v

    # Springls: the exact outline resampled at the arm spacing (one springl per sample interval, the
    # particle at its middle), then the flow's small error: a slight jitter, and two missing.
    samples = resample(outline, 0.9 * h_t)
    rng = Lcg(7)
    springls = []
    for k in range(len(samples)):
        a, b = samples[k], samples[(k + 1) % len(samples)]
        p = lerp(a, b, 0.5)
        jitter = (rng.uniform(-0.03, 0.03) * h_t, rng.uniform(-0.03, 0.03) * h_t)
        ang = rng.uniform(-0.03, 0.03)
        def tr(q):
            d = sub(q, p)
            c, s = math.cos(ang), math.sin(ang)
            return add(add(p, jitter), (c * d[0] - s * d[1], s * d[0] + c * d[1]))
        springls.append((tr(a), tr(b), add(p, jitter)))
    # Open a gap: remove the springls nearest a point on the top tab's flank.
    gap_at = (0.38 * R, 1.25 * R)
    springls.sort(key=lambda s: norm(sub(s[2], gap_at)))
    removed = springls[:2]
    springls = springls[2:]

    # 1. U per vertex (exact: the patch is flat).
    U = [min(segment_distance(p, s[0], s[1]) for s in springls) for p in V]
    wall = [u <= capsule for u in U]
    # Vertex adjacency over the conforming mesh.
    adj = [set() for _ in V]
    for (a, b, c), _, _ in faces:
        adj[a] |= {b, c}
        adj[b] |= {a, c}
        adj[c] |= {a, b}
    def side(k):
        """Side of vertex k of its nearest springl (the region on the springls' left): -1 inside, +1 outside."""
        p = V[k]
        ranked = sorted(springls, key=lambda sp: segment_distance(p, sp[0], sp[1]))
        a, b = ranked[0][0], ranked[0][1]
        ab = sub(b, a)
        t = dot(sub(p, a), ab) / dot(ab, ab)
        left = lambda u, v: unit((-(v[1] - u[1]), v[0] - u[0]))
        n = left(a, b)
        foot = add(a, mul(ab, max(0.0, min(1.0, t))))
        if t <= 0.0 or t >= 1.0:
            # Nearest an arm's outer end: the pseudo-normal with the neighboring springl, as at a polyline vertex.
            end = a if t <= 0.0 else b
            other = min(ranked[1:6], key=lambda sp: min(norm(sub(sp[0], end)), norm(sub(sp[1], end))))
            n = add(n, left(other[0], other[1]))
        return -1 if dot(sub(p, foot), n) > 0 else 1

    # 2. Exterior flood from the non-wall vertex with the largest U.
    seed = max((u, -k) for k, u in enumerate(U) if not wall[k])
    seed = -seed[1]
    sign = [0] * len(V)
    cls = [None] * len(V)
    q = deque([seed])
    sign[seed] = 1
    while q:
        k = q.popleft()
        for n in adj[k]:
            if not wall[n] and sign[n] == 0:
                sign[n] = 1
                q.append(n)
    for k in range(len(V)):
        if not wall[k] and sign[k] == 1:
            cls[k] = FLOOD_OUT
    # 2a. Each region the flood did not reach takes the side its bordering wall vertexes vote for
    # (inside on a tie): an exterior pocket whose mouth the walls close is voted outside.
    for k0 in range(len(V)):
        if wall[k0] or sign[k0] != 0:
            continue
        region, q = [k0], deque([k0])
        sign[k0] = 2
        border = set()
        while q:
            k = q.popleft()
            for n in adj[k]:
                if wall[n]:
                    border.add(n)
                elif sign[n] == 0:
                    sign[n] = 2
                    region.append(n)
                    q.append(n)
        vote = sum(side(b) for b in border)
        s_region = 1 if vote > 0 else -1
        for k in region:
            sign[k] = s_region
            cls[k] = POCKET_OUT if s_region > 0 else UNREACHED_IN
    # 3. Upwind sweep into the walls; a local maximum of U takes its springl side.
    for k in sorted((k for k in range(len(V)) if wall[k]), key=lambda k: (-U[k], k)):
        vote = sum((U[n] - U[k]) * sign[n] for n in adj[k] if U[n] > U[k] and sign[n] != 0)
        if vote != 0:
            sign[k] = 1 if vote > 0 else -1
            cls[k] = WALL_OUT if vote > 0 else WALL_IN
        else:
            sign[k] = side(k)
            cls[k] = FALLBACK_IN
    # 4. Vertexes within 1 h_t of a springl take its side.
    for k in range(len(V)):
        if U[k] <= 1.0 * h_t:
            sd = side(k)
            if sd != sign[k]:
                sign[k] = sd
                cls[k] = CORRECTED
    clamp = 3.5 * h_t
    phi = [sign[k] * min(U[k], clamp) for k in range(len(V))]
    contour, crossing = refine_contour(marching_triangles(faces, V, phi, with_edges=True), V, springls, 2.0 * h_t)

    W, H = 1140, 1080
    svg = Svg(W, H)
    pw, ph, gap = 555, 515, 10
    boxes = [(gap, 20, pw, ph), (2 * gap + pw, 20, pw, ph), (gap, 30 + ph, pw, ph), (2 * gap + pw, 30 + ph, pw, ph)]
    view = (-1.75 * R, -1.65 * R, 1.75 * R, 1.85 * R)

    def mapper(box):
        return fit2d([(view[0], view[1]), (view[2], view[3])], (box[0], box[1] + 34, box[2], box[3] - 80), margin=0.02)

    def clip_open(box, to):
        x0, y0 = to((view[0], view[3]))
        x1, y1 = to((view[2], view[1]))
        cid = f"c{box[0]}_{box[1]}"
        svg.add(f'<clipPath id="{cid}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="4"/></clipPath>')
        svg.add(f'<g clip-path="url(#{cid})">')

    def draw_mesh(to, fill=None, opacity=1.0, edge_color="#3A3D4A", width=0.7):
        for (a, b, c), lvl, green in faces:
            pts = [to(V[a]), to(V[b]), to(V[c])]
            if fill is not None:
                svg.polygon(pts, fill((a, b, c), lvl, green), opacity)
            svg.polyline(pts, edge_color, width, closed=True)

    def draw_springls(to, faded=False):
        for a, b, p in springls:
            svg.line(to(a), to(b), SPRINGL, 2.2, opacity=0.6 if faded else 1.0)
        if not faded:
            for a, b, p in springls:
                svg.circle(to(p), 1.9, PARTICLE)
                svg.circle(to(a), 1.4, VERTEX)
                svg.circle(to(b), 1.4, VERTEX)

    # (a) springls and the refined mesh.
    box = boxes[0]
    svg.panel(*box, "(a) Springls after a sub-step; the mesh refined near them")
    to = mapper(box)
    clip_open(box, to)
    draw_mesh(to, fill=lambda f, lvl, green: LEVELS[min(lvl, len(LEVELS) - 1)], opacity=0.42, edge_color="#4A4E60")
    draw_springls(to)
    for a, b, p in removed:
        svg.line(to(a), to(b), PARTICLE, 1.2, dash="3 3", opacity=0.9)
    svg.add("</g>")
    g = to(gap_at)
    svg.add(f'<circle cx="{g[0]:.1f}" cy="{g[1]:.1f}" r="16" fill="none" stroke="{PARTICLE_LABEL}" stroke-width="1.4"/>')
    svg.text(g[0] + 22, g[1] - 14, "gap: two springls missing", 13, TEXT, weight=500, halo=True)
    legend(svg, box, [(LEVELS[0], "input faces"), (LEVELS[1], "level 1"), (LEVELS[2], "level 2"), ("#8A93A6", "springls")])

    # (b) U and the walls.
    box = boxes[1]
    svg.panel(*box, "(b) U, the distance to the springls; walls U ≤ capsule")
    to = mapper(box)
    clip_open(box, to)
    umax = 2.2 * R

    def ufill(f, lvl, green):
        u = sum(min(U[v], umax) for v in f) / 3 / umax
        c = int(30 + 150 * (1 - u) ** 2)
        return "#%02X%02X%02X" % (c // 2, c // 2 + 8, c)

    draw_mesh(to, fill=ufill, opacity=1.0, edge_color="#2C2E38", width=0.5)
    for k, p in enumerate(V):
        if wall[k] and view[0] <= p[0] <= view[2] and view[1] <= p[1] <= view[3]:
            svg.circle(to(p), 2.1, CONTOUR)
    draw_springls(to, faded=True)
    svg.add("</g>")
    legend(svg, box, [(CONTOUR, "wall vertexes (U ≤ 1.5 h<tspan baseline-shift='sub' font-size='9'>t</tspan>)"), ("#5A6496", "U small"), ("#0F1018", "U large")])

    # (c) the sign.
    box = boxes[2]
    svg.panel(*box, "(c) Sign: flood, side vote, upwind sweep")
    to = mapper(box)
    clip_open(box, to)

    def sfill(f, lvl, green):
        # Majority class of the face's vertexes, for a readable picture.
        cs = [cls[v] for v in f]
        return max(set(cs), key=cs.count)

    draw_mesh(to, fill=sfill, opacity=1.0, edge_color="#1B1C22", width=0.45)
    draw_springls(to, faded=True)
    svg.add("</g>")
    items = [(FLOOD_OUT, "flooded: outside"), (UNREACHED_IN, "unreached, voted inside"), (POCKET_OUT, "unreached, voted outside"),
             (WALL_OUT, "wall, swept outside"), (WALL_IN, "wall, swept inside"), (FALLBACK_IN, "wall, local max of U"),
             (CORRECTED, "corrected to springl side")]
    legend(svg, box, [it for it in items if it[0] in cls])  # only the classes that occur

    # (d) phi and the contour.
    box = boxes[3]
    svg.panel(*box, "(d) φ = sign · min(U, 3.5 h<tspan baseline-shift='sub' font-size='10'>t</tspan>) and its contour")
    to = mapper(box)
    clip_open(box, to)

    def pfill(f, lvl, green):
        v = min(sum(phi[k] for k in f) / 3 / clamp, -0.05)
        if v < 0:
            t = min(-v, 1)
            base = (0xE8, 0xE4, 0xDA)
            return "#%02X%02X%02X" % tuple(int(40 + (c - 40) * (0.35 + 0.65 * t)) for c in base)
        t = min(v, 1)
        return "#%02X%02X%02X" % (int(34 + 20 * (1 - t)), int(36 + 24 * (1 - t)), int(44 + 40 * (1 - t)))

    # Outside shading on every face, then each face's part with phi < 0 (clipped at the refined contour
    # crossings) in the region's shading, so the fill ends exactly on the contour.
    def outside_fill(f, lvl, green):
        t = min(max(sum(phi[k] for k in f) / 3 / clamp, 0.0), 1.0)
        return "#%02X%02X%02X" % (int(34 + 20 * (1 - t)), int(36 + 24 * (1 - t)), int(44 + 40 * (1 - t)))

    for (a, b, c), lvl, green in faces:
        svg.polygon([to(V[a]), to(V[b]), to(V[c])], outside_fill((a, b, c), lvl, green))
        poly = []
        for p, q in [(a, b), (b, c), (c, a)]:
            if phi[p] < 0:
                poly.append(V[p])
            if (phi[p] < 0) != (phi[q] < 0):
                poly.append(lerp(V[p], V[q], crossing[(p, q)]))
        if len(poly) >= 3:
            svg.polygon([to(x) for x in poly], pfill((a, b, c), lvl, green))
    draw_mesh(to, fill=None, edge_color="#2A2B33", width=0.45)
    for a, b in contour:
        svg.line(to(a), to(b), CONTOUR, 2.6)
    draw_springls(to)
    svg.add("</g>")
    g = to(gap_at)
    svg.add(f'<circle cx="{g[0]:.1f}" cy="{g[1]:.1f}" r="16" fill="none" stroke="{CONTOUR}" stroke-width="1.4"/>')
    svg.text(g[0] + 22, g[1] - 14, "contour bridges the gap", 13, TEXT, weight=500, halo=True)
    legend(svg, box, [(OBJECT, "φ &lt; 0 (region)"), (CONTOUR, "contour (marching triangles)"), (PARTICLE, "springl particles")])
    svg.save("intrinsic-reconstruction.svg")


def legend(svg, box, items):
    x, y = box[0] + 14, box[1] + box[3] - 38
    for color, label in items:
        svg.add(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="2" fill="{color}" stroke="#55586A" stroke-width="0.6"/>')
        svg.text(x + 17, y, label, 12, DIM_TEXT)
        x += 24 + 6.4 * len(label.split("<")[0]) + (14 if "<" in label else 0)
        if x > box[0] + box[2] - 120:
            x, y = box[0] + 14, y + 18


def segment_intersection(a, b, c, d):
    """Parameter along a -> b where it crosses segment c -> d, or None."""
    r, q = sub(b, a), sub(d, c)
    den = cross2(r, q)
    if abs(den) < 1e-15:
        return None
    t = cross2(sub(c, a), q) / den
    u = cross2(sub(c, a), r) / den
    return t if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


def refine_contour(segs, V, springls, proximity, iterations=4, step=0.5):
    """As IntrinsicSpringls::refineContour: each crossing slides along its own edge toward the nearest
    springl particle within `proximity` (weight 1 - distance / proximity), then a crossing on an edge that
    a springl arm crosses is placed on that arm's crossing."""
    out, crossing = [], {}
    for seg in segs:
        pts = []
        for (p, q), t in seg:
            if (p, q) in crossing:
                pts.append(lerp(V[p], V[q], crossing[(p, q)]))
                continue
            a, b = V[p], V[q]
            for _ in range(iterations):
                x = lerp(a, b, t)
                near = min(springls, key=lambda sp: norm(sub(sp[2], x)))
                dist = norm(sub(near[2], x))
                if dist < proximity:
                    ab = sub(b, a)
                    target = max(0.0, min(1.0, dot(sub(near[2], a), ab) / dot(ab, ab)))
                    t += step * (1 - dist / proximity) * (target - t)
            hits = [segment_intersection(a, b, sp[0], sp[2]) for sp in springls] + \
                   [segment_intersection(a, b, sp[2], sp[1]) for sp in springls]
            hits = [h for h in hits if h is not None]
            if hits:
                t = min(hits, key=lambda h: abs(h - t))
            crossing[(p, q)] = t
            crossing[(q, p)] = 1.0 - t
            pts.append(lerp(a, b, t))
        out.append((pts[0], pts[1]))
    return out, crossing


def marching_triangles(faces, V, phi, with_edges=False):
    segs = []
    for (a, b, c), _, _ in faces:
        pts = []
        for p, q in [(a, b), (b, c), (c, a)]:
            if (phi[p] < 0) != (phi[q] < 0):
                t = phi[p] / (phi[p] - phi[q])
                pts.append(((p, q), t) if with_edges else lerp(V[p], V[q], t))
        if len(pts) == 2 and with_edges:
            segs.append((pts[0], pts[1]))
            continue
        if len(pts) == 2:
            # Orient so the region (phi < 0) is on the left.
            inside = [V[k] for k in (a, b, c) if phi[k] < 0]
            ci = mul(tuple(map(sum, zip(*inside))), 1.0 / len(inside))
            if cross2(sub(pts[1], pts[0]), sub(ci, pts[0])) < 0:
                pts.reverse()
            segs.append((pts[0], pts[1]))
    return segs


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def uniform(self, a, b):
        self.s = (1103515245 * self.s + 12345) % (1 << 31)
        return a + (b - a) * self.s / (1 << 31)


if __name__ == "__main__":
    figure_flip()
    figure_trace()
    figure_reconstruction()
