#!/usr/bin/env python3
"""Generates docs/springls-anatomy.svg: the anatomy of a springl, and spring level sets vs. doubleweave.

Colors match the enright2d panels. Run from anywhere: python3 docs/make_anatomy_diagram.py
"""
import math
from pathlib import Path

# Palette (same as examples/enright2d/Renderer.cpp).
BACKGROUND = "#202024"
PANEL = "#141418"
GRID = "#464652"
TEXT = "#e6e6e6"
DIM_TEXT = "#a8a8b0"
CONTOUR = "#008080"
CONTOUR_FILL = "rgba(0,128,128,0.45)"
LEVEL_LINE = "#3c8c8c"
PRIMAL_SPRINGL = "#9a9a9a"
PRIMAL_PARTICLE = "#c01010"
DUAL_SPRINGL = "#ff8c00"
DUAL_PARTICLE = "#2a85ff"
VERTEX = "#ffcc00"
NORMAL = "#33cc33"
PRIMAL_CAPSULE = "rgba(236,64,122,0.35)"
DUAL_CAPSULE = "rgba(64,160,236,0.35)"
FONT = "Helvetica, Arial, sans-serif"

WIDTH, HEIGHT = 980, 830
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
    emit(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
         f'font-weight="{weight}"{style}>{s}</text>')


def line(a, b, color, width, dash=None, cap="round", opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{opacity}"' if opacity is not None else ""
    emit(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}" '
         f'stroke-width="{width}" stroke-linecap="{cap}"{d}{o}/>'.replace(f'stroke="{color}"', paint("stroke", color)))


def circle(c, r, fill, stroke=None, width=0):
    s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    emit(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r}" {paint("fill", fill)}{s}/>')


def polyline(points, color, width, fill="none", dash=None, closed=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    tag = "polygon" if closed else "polyline"
    emit(f'<{tag} points="{pts}" {paint("fill", fill)} {paint("stroke", color)} stroke-width="{width}" '
         f'stroke-linejoin="round" stroke-linecap="round"{d}/>')


def arrow(a, b, color, width=2.5, head=9):
    line(a, b, color, width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45))
    p2 = (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))
    polyline([p1, b, p2], color, width)


def leader(frm, to, label, anchor="start", color=DIM_TEXT, dy=4):
    """Label at `frm` with a thin line to the feature at `to`."""
    line(frm, to, color, 1)
    circle(to, 2, color)
    off = 4 if anchor == "start" else -4
    text(frm[0] + off, frm[1] + dy, label, 13, TEXT, anchor)


def spring(a, b, coils=6, amp=6, color=DIM_TEXT):
    """Zigzag spring from a to b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    pts = [a]
    n = coils * 2
    for k in range(1, n):
        t = k / n
        s = amp if k % 2 else -amp
        pts.append((a[0] + dx * t + nx * s, a[1] + dy * t + ny * s))
    pts.append(b)
    polyline(pts, color, 1.6)


def springl(p, tangent, half, particle_color, springl_color, normal_len, sizes=(5, 3.5), width=3,
            capsule=None, capsule_radius=0):
    """Draws a springl: capsule, arms, normal, vertexes and particle."""
    tx, ty = tangent
    nx, ny = ty, -tx  # outward normal (left of travel, inside on the right)
    v0 = (p[0] - tx * half, p[1] - ty * half)
    v1 = (p[0] + tx * half, p[1] + ty * half)
    if capsule:
        line(v0, v1, capsule, 2 * capsule_radius)
    line(v0, v1, springl_color, width)
    arrow(p, (p[0] + nx * normal_len, p[1] + ny * normal_len), NORMAL, 2, 6)
    circle(v0, sizes[1], VERTEX)
    circle(v1, sizes[1], VERTEX)
    circle(p, sizes[0], particle_color)
    return v0, v1


# ------------------------------------------------------------------------------------------------
emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
     f'font-family="{FONT}" role="img" aria-labelledby="title">')
emit('<title id="title">Anatomy of a springl, and spring level sets vs. doubleweave</title>')
emit(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>')

# ---- (a) Anatomy of one springl -----------------------------------------------------------------
AX, AY, AW, AH = 20, 20, 940, 270
emit(f'<rect x="{AX}" y="{AY}" width="{AW}" height="{AH}" rx="6" fill="{PANEL}"/>')
text(AX + 16, AY + 28, "(a) Anatomy of a springl", 18, TEXT, weight="bold")

p = (AX + 250, AY + 150)
v0, v1 = springl(p, (1.0, 0.0), 120, PRIMAL_PARTICLE, PRIMAL_SPRINGL, 70, sizes=(9, 7), width=5,
                 capsule=PRIMAL_CAPSULE, capsule_radius=34)
# A neighboring springl to the right; its start vertex is attracted to this springl's end vertex.
ang = math.radians(14)
t2 = (math.cos(ang), math.sin(ang))
n0_target = (v1[0] + 70, v1[1] + 16)
p2 = (n0_target[0] + 95 * t2[0], n0_target[1] + 95 * t2[1])
n0, _ = springl(p2, t2, 95, PRIMAL_PARTICLE, PRIMAL_SPRINGL, 55, sizes=(8, 6), width=4)
spring(v1, n0, coils=4, amp=6)

leader((p[0] - 40, AY + 64), (p[0] - 70, p[1] - 33), "capsule (radius = extent)", anchor="end")
leader((p[0] + 22, AY + 72), (p[0], p[1] - 52), "normal n (perpendicular to v₀ → v₁)")
leader((v0[0] - 30, p[1] + 52), v0, "vertex v₀", anchor="end")
leader((p[0] - 30, AY + 232), (p[0] - 60, p[1] + 2), "arm (spring to rest length)", anchor="end")
leader((p[0] + 20, AY + 258), (p[0] + 2, p[1] + 9), "particle p (advected by the flow)")
leader((v1[0] + 16, AY + 228), v1, "vertex v₁")
leader((n0[0] + 40, AY + 254), ((v1[0] + n0[0]) / 2 + 4, (v1[1] + n0[1]) / 2 + 5), "attraction to the nearest neighbor vertex")
text(p2[0] + 28, p2[1] - 36, "neighbor springl", 13, DIM_TEXT)

for k, line_text in enumerate([
    "A springl is a short line segment: a",
    "particle with two arms ending in",
    "vertexes, and a normal. Particles are",
    "advected by the flow; the arms relax",
    "under springs, keeping the springl's",
    "length and turning it toward its",
    "neighbors' vertexes.",
]):
    text(AX + 700, AY + 64 + 19 * k, line_text, 13, DIM_TEXT)


# ---- (b), (c): interface on a level-set grid ----------------------------------------------------
PW, PH = 460, 470
CELL = 64
R = 400.0  # interface: arc of a circle, inside (phi < 0) below it
PEAK = 120  # arc peak below the grid top
STEP = CELL / R  # one cell of arc length between springls of a constellation
HALF = 22.0  # springl half-length: shorter than half the spacing, so one constellation has gaps


def scene(ox, oy, title, subtitle):
    """Draws the grid, level set and iso-contour; returns geometry helpers."""
    emit(f'<rect x="{ox}" y="{oy}" width="{PW}" height="{PH}" rx="6" fill="{PANEL}"/>')
    text(ox + 16, oy + 28, title, 18, TEXT, weight="bold")
    text(ox + 16, oy + 48, subtitle, 13, DIM_TEXT)
    gx, gy, gw, gh = ox + 10, oy + 70, PW - 20, 320
    clip = f"clip{ox}"
    emit(f'<clipPath id="{clip}"><rect x="{gx}" y="{gy}" width="{gw}" height="{gh}"/></clipPath>')
    emit(f'<g clip-path="url(#{clip})">')
    cx, cy = gx + gw / 2, gy + PEAK + R

    def arc_points(radius, step=0.005):
        pts, a = [], -1.3
        while a <= 1.3:
            pts.append((cx + radius * math.sin(a), cy - radius * math.cos(a)))
            a += step
        return pts

    cols, rows = int(gw / CELL) + 1, int(gh / CELL) + 1
    # Iso-contour: marching-squares vertexes where the interface crosses grid lines.
    verts = []
    for i in range(cols + 1):
        x = gx + i * CELL
        d = R * R - (x - cx) ** 2
        if d > 0:
            verts.append((x, cy - math.sqrt(d)))
    for j in range(rows + 1):
        y = gy + j * CELL
        d = R * R - (y - cy) ** 2
        if d > 0:
            verts += [(cx - math.sqrt(d), y), (cx + math.sqrt(d), y)]
    verts = [v for v in verts if gy - 1 <= v[1] <= gy + gh + 1]
    verts.sort(key=lambda v: math.atan2(v[0] - cx, cy - v[1]))
    # Inside of the level set, grid, distance lines, contour.
    polyline(verts + [(gx + gw + 80, gy + gh + 80), (gx - 80, gy + gh + 80)], "none", 0, fill=CONTOUR_FILL, closed=True)
    for i in range(cols + 1):
        line((gx + i * CELL, gy), (gx + i * CELL, gy + gh), GRID, 1, cap="butt")
    for j in range(rows + 1):
        line((gx, gy + j * CELL), (gx + gw, gy + j * CELL), GRID, 1, cap="butt")
    for k in (-2, -1, 1, 2):  # phi = k cells: radius R + k * CELL (phi > 0 outside)
        polyline(arc_points(R + k * CELL), LEVEL_LINE, 1.3, dash="6,5")
    polyline(verts, CONTOUR, 4)
    for v in verts:
        circle(v, 3.5, CONTOUR, "#bfffff", 1.2)
    emit("</g>")

    def on_arc(theta, radius=R):
        return (cx + radius * math.sin(theta), cy - radius * math.cos(theta)), (math.cos(theta), math.sin(theta))

    # Distance labels for the outside lines, at the left edge just above each line.
    for k in (1, 2):
        x = gx + 6
        y = cy - math.sqrt((R + k * CELL) ** 2 - (x - cx) ** 2)
        if y - 6 > gy + 14:
            text(x, y - 6, f"φ = +{k}", 12, "#6fc8c8")
    return dict(gx=gx, gy=gy, gw=gw, gh=gh, cx=cx, cy=cy, on_arc=on_arc, verts=verts, clip=clip)


def draw_constellation(s, offset, particle_color, springl_color):
    emit(f'<g clip-path="url(#{s["clip"]})">')
    for k in range(-8, 9):
        pt, tan = s["on_arc"]((k + offset) * STEP)
        springl(pt, tan, HALF, particle_color, springl_color, 30, sizes=(6.5, 4.5), width=4)
    emit("</g>")


def scene_labels(s):
    gx, gy, gw, gh = s["gx"], s["gy"], s["gw"], s["gh"]
    pt, _ = s["on_arc"](2.6 * STEP)
    leader((gx + gw - 8, gy + 24), (pt[0], pt[1] - 1), "iso-contour φ = 0", anchor="end")
    text(gx + 8, gy + gh - 12, "inside: φ &lt; 0", 13, TEXT)
    text(gx + gw - 8, gy + gh - 12, "level-set grid", 13, DIM_TEXT, anchor="end")


# (b) Spring level set.
BX, BY = 20, 310
sb = scene(BX, BY, "(b) Spring level set", "one constellation; the iso-contour makes the surface watertight")
draw_constellation(sb, 0.0, PRIMAL_PARTICLE, PRIMAL_SPRINGL)
scene_labels(sb)
gp, _ = sb["on_arc"](0.5 * STEP)
leader((gp[0] + 34, sb["gy"] + 222), (gp[0], gp[1] + 3), "gap between springls")
left = [v for v in sb["verts"] if v[0] < sb["cx"] - 120]
pv = left[-1]
leader((sb["gx"] + 16, sb["gy"] + 252), pv, "contour vertex on a grid edge")
text(BX + 16, BY + PH - 50, "Springls track the interface; the level set is evolved toward", 13, DIM_TEXT)
text(BX + 16, BY + PH - 32, "them, and its iso-contour closes the gaps between them. Springls", 13, DIM_TEXT)
text(BX + 16, BY + PH - 14, "far from the contour are removed; uncovered edges are filled.", 13, DIM_TEXT)

# (c) Doubleweave.
CX, CY = 500, 310
sc = scene(CX, CY, "(c) Doubleweave", "two staggered constellations that cover each other's gaps")
draw_constellation(sc, 0.0, PRIMAL_PARTICLE, PRIMAL_SPRINGL)
draw_constellation(sc, 0.5, DUAL_PARTICLE, DUAL_SPRINGL)
scene_labels(sc)
pp, _ = sc["on_arc"](-1.0 * STEP)
leader((sc["gx"] + 16, sc["gy"] + 222), (pp[0], pp[1] + 6), "primal springl (contour edge)")
dp, _ = sc["on_arc"](0.5 * STEP)
leader((sc["gx"] + sc["gw"] - 16, sc["gy"] + 252), (dp[0], dp[1] + 6), "dual springl covers the primal gap", anchor="end")
text(CX + 16, CY + PH - 50, "Primal (red / gray) and dual (blue / orange) springls are offset", 13, DIM_TEXT)
text(CX + 16, CY + PH - 32, "by half an edge and overlap, so their union is watertight. Each", 13, DIM_TEXT)
text(CX + 16, CY + PH - 14, "constellation is filled and contracted against its own level set.", 13, DIM_TEXT)

# ---- Legend --------------------------------------------------------------------------------------
LY = HEIGHT - 26
x = 28
for kind, label in [("particle", "primal particle"), ("dual", "dual particle"), ("vertex", "vertex"),
                    ("normal", "normal"), ("capsule", "capsule"), ("contour", "iso-contour"),
                    ("level", "level-set distance (cells)"), ("grid", "grid cell")]:
    if kind == "particle":
        circle((x + 7, LY), 6, PRIMAL_PARTICLE)
    elif kind == "dual":
        circle((x + 7, LY), 6, DUAL_PARTICLE)
    elif kind == "vertex":
        circle((x + 7, LY), 5, VERTEX)
    elif kind == "normal":
        arrow((x + 2, LY + 6), (x + 14, LY - 6), NORMAL, 2, 5)
    elif kind == "capsule":
        line((x + 4, LY), (x + 12, LY), PRIMAL_CAPSULE, 12)
    elif kind == "contour":
        line((x, LY), (x + 16, LY), CONTOUR, 4)
    elif kind == "level":
        line((x, LY), (x + 16, LY), LEVEL_LINE, 1.5, dash="4,3", cap="butt")
    elif kind == "grid":
        emit(f'<rect x="{x}" y="{LY - 7}" width="14" height="14" fill="none" stroke="{GRID}" stroke-width="1.5"/>')
    text(x + 22, LY + 4.5, label, 12, TEXT)
    x += 22 + 6.4 * len(label) + 16

emit("</svg>")
path = Path(__file__).resolve().parent / "springls-anatomy.svg"
path.write_text("\n".join(out) + "\n")
print(f"wrote {path}  (legend ends at x = {x:.0f} of {WIDTH})")
