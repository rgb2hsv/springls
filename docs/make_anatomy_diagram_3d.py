#!/usr/bin/env python3
"""Generates docs/springls-anatomy-3d.svg: the anatomy of a 3D springl, and the primal and dual
(doubleweave) springls on an iso-surface.

Colors follow the enright3d panels. Run from anywhere: python3 docs/make_anatomy_diagram_3d.py
"""
import math
from pathlib import Path

# Palette (examples/enright3d/Renderer.cpp, particles brightened to read on a flat diagram).
BACKGROUND = "#202024"
PANEL = "#141418"
TEXT = "#e6e6e6"
DIM_TEXT = "#a8a8b0"
SURFACE = "rgba(200,200,208,0.10)"
SURFACE_EDGE = "#8a8a94"
PRIMAL = "rgba(158,82,77,0.85)"
PRIMAL_EDGE = "#d08078"
PRIMAL_PARTICLE = "#ff6a5c"
DUAL = "rgba(77,107,158,0.85)"
DUAL_EDGE = "#80a4e0"
DUAL_PARTICLE = "#6aa0ff"
VERTEX = "#ffcc00"
NORMAL = "#33cc33"
CAPSULE = "rgba(97,138,97,0.35)"
FONT = "Helvetica, Arial, sans-serif"

WIDTH, HEIGHT = 980, 920
out = []


def emit(s):
    out.append(s)


def paint(attr, color):
    """attr="color" for hex colors; attr + attr-opacity for rgba(...), which not every SVG renderer accepts."""
    if color.startswith("rgba("):
        r, g, b, a = (float(v) for v in color[5:-1].split(","))
        return f'{attr}="#{int(r):02x}{int(g):02x}{int(b):02x}" {attr}-opacity="{a}"'
    return f'{attr}="{color}"'


def text(x, y, s, size=14, color=TEXT, anchor="start", weight="normal"):
    emit(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
         f'font-weight="{weight}">{s}</text>')


def paragraph(x, y, lines, size=13, color=DIM_TEXT, spacing=19):
    for k, s in enumerate(lines):
        text(x, y + k * spacing, s, size, color)


def line(a, b, color, width, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    emit(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" {paint("stroke", color)} '
         f'stroke-width="{width}" stroke-linecap="round"{d}/>')


def circle(c, r, fill):
    emit(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r}" {paint("fill", fill)}/>')


def polygon(points, fill, stroke="none", width=1, closed=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    tag = "polygon" if closed else "polyline"
    emit(f'<{tag} points="{pts}" {paint("fill", fill)} {paint("stroke", stroke)} stroke-width="{width}" '
         f'stroke-linejoin="round" stroke-linecap="round"{d}/>')


def arrow(a, b, color, width=2.5, head=9):
    line(a, b, color, width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45))
    p2 = (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))
    polygon([p1, b, p2], "none", color, width, closed=False)


def leader(frm, to, label, anchor="start"):
    """Label at `frm` with a thin line to the feature at `to`."""
    line(frm, to, DIM_TEXT, 1)
    circle(to, 2, DIM_TEXT)
    off = 4 if anchor == "start" else -4
    text(frm[0] + off, frm[1] + 4, label, 13, TEXT, anchor)


def spring(a, b, coils=5, amp=5):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    pts = [a]
    n = coils * 2
    for k in range(1, n):
        t = k / n
        s = amp if k % 2 else -amp
        pts.append((a[0] + dx * t + nx * s, a[1] + dy * t + ny * s))
    pts.append(b)
    polygon(pts, "none", DIM_TEXT, 1.6, closed=False)


# ---- 3D helpers ---------------------------------------------------------------------------------
def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, s):
    return tuple(x * s for x in a)


def centroid(points):
    return mul(tuple(map(sum, zip(*points))), 1.0 / len(points))


def lerp(a, b, t):
    return add(a, mul(sub(b, a), t))


class View:
    """Orthographic view from the front, above: azimuth about z, then elevation."""

    def __init__(self, origin, scale, azimuth=-28.0, elevation=38.0):
        self.origin, self.scale = origin, scale
        self.ca, self.sa = math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))
        self.ce, self.se = math.cos(math.radians(elevation)), math.sin(math.radians(elevation))

    def rotate(self, p):
        x, y, z = p
        return x * self.ca - y * self.sa, x * self.sa + y * self.ca, z

    def __call__(self, p):
        xr, yr, z = self.rotate(p)
        return (self.origin[0] + self.scale * xr, self.origin[1] - self.scale * (yr * self.se + z * self.ce))

    def depth(self, p):  # larger is farther from the viewer
        _, yr, z = self.rotate(p)
        return yr * self.ce - z * self.se


# ------------------------------------------------------------------------------------------------
emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
     f'font-family="{FONT}" role="img" aria-labelledby="title">')
emit('<title id="title">Anatomy of a 3D springl, and primal and dual springls on an iso-surface</title>')
emit(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>')

# ---- (a) Anatomy of one 3D springl --------------------------------------------------------------
AX, AY, AW, AH = 20, 20, 940, 400
emit(f'<rect x="{AX}" y="{AY}" width="{AW}" height="{AH}" rx="6" fill="{PANEL}"/>')
text(AX + 16, AY + 28, "(a) Anatomy of a 3D springl", 18, TEXT, weight="bold")

view = View((AX + 215, AY + 240), 78)
p = (0.0, 0.0, 0.0)
quad = [(-1.0, -0.9, 0.0), (1.0, -1.0, 0.0), (1.05, 0.95, 0.0), (-0.95, 1.0, 0.0)]
# Neighbor springl beyond the right edge, tilted slightly and offset by about half a springl, so
# each vertex's nearest neighbor feature is a point inside an edge, not a vertex.
tilt = 0.25
nquad = [(1.45, -0.2, 0.12), (3.2, -0.1, 0.12 + tilt * 1.75), (3.15, 1.8, 0.12 + tilt * 1.7), (1.4, 1.75, 0.1)]
np_ = centroid(nquad)


def closest_on_segment(q, a, b):
    ab = sub(b, a)
    t = sum(x * y for x, y in zip(sub(q, a), ab)) / sum(x * x for x in ab)
    return lerp(a, b, min(1.0, max(0.0, t)))
capsule_r = 0.42

screen = [view(v) for v in quad]
# Capsule: the Minkowski sum of the polygon and a sphere projects to the polygon offset by the radius.
polygon(screen, CAPSULE, CAPSULE, 2 * capsule_r * view.scale)
polygon(screen, PRIMAL, PRIMAL_EDGE, 2)
for v in quad:
    line(view(p), view(v), "#f0d0c8", 2, dash="5 4")
# Neighbor springl.
nscreen = [view(v) for v in nquad]
polygon(nscreen, PRIMAL, PRIMAL_EDGE, 1.5)
for v in nquad:
    line(view(np_), view(v), "#f0d0c8", 1.2, dash="4 4")
for v in nscreen:
    circle(v, 4.5, VERTEX)
circle(view(np_), 6, PRIMAL_PARTICLE)
# Attraction: each vertex is pulled toward the closest point on the nearby edges of other springls.
pull_a = closest_on_segment(quad[2], nquad[3], nquad[0])  # this springl's vertex -> neighbor edge
pull_b = closest_on_segment(nquad[0], quad[1], quad[2])  # neighbor's vertex -> this springl's edge
for vtx, target in ((quad[2], pull_a), (nquad[0], pull_b)):
    spring(view(vtx), view(target), coils=3, amp=4)
    emit(f'<circle cx="{view(target)[0]:.1f}" cy="{view(target)[1]:.1f}" r="4" fill="none" stroke="{TEXT}" stroke-width="1.5"/>')
# Normal, vertexes, particle.
arrow(view(p), view((0, 0, 1.25)), NORMAL, 2.5, 9)
for v in screen:
    circle(v, 6.5, VERTEX)
circle(view(p), 8.5, PRIMAL_PARTICLE)

leader((AX + 150, AY + 62), view((0.0, 0.0, 1.0)), "normal n (fan normal about p)", "start")
leader((AX + 40, AY + 120), view(quad[3]), "vertex v₃", "start")
leader((AX + 30, AY + 330), view(quad[0]), "vertex v₀", "start")
leader((AX + 120, AY + 362), lerp(view(p), view(quad[0]), 0.55), "arm (particle to vertex)", "start")
leader((AX + 290, AY + 350), view(p), "particle p (advected by the flow)", "start")
leader((AX + 30, AY + 230), (view(quad[0])[0] - 0.42 * view.scale + 4, view(quad[0])[1] - 8), "capsule", "start")
leader((AX + 360, AY + 92), lerp(view(quad[2]), view(pull_a), 0.5), "attraction: vertex to the nearest", "start")
text(AX + 364, AY + 110, "point on another springl's edge", 13, TEXT)
line((AX + 360, AY + 96), lerp(view(nquad[0]), view(pull_b), 0.5), DIM_TEXT, 1)
circle(lerp(view(nquad[0]), view(pull_b), 0.5), 2, DIM_TEXT)
leader((AX + 470, AY + 160), view(np_), "neighbor springl", "start")


# Element types: flat drawings of the three springl shapes.
def flat_springl(center, radius, n, fill, edge, particle, phase=0.0, squash=0.62):
    pts = [(center[0] + radius * math.cos(phase + 2 * math.pi * k / n),
            center[1] - squash * radius * math.sin(phase + 2 * math.pi * k / n)) for k in range(n)]
    polygon(pts, fill, edge, 1.5)
    for v in pts:
        line(center, v, "#f0d0c8", 1, dash="3 3")
        circle(v, 3.5, VERTEX)
    circle(center, 5, particle)


EX = AX + 690
text(EX, AY + 262, "Element types", 14, TEXT, weight="bold")
flat_springl((EX + 30, AY + 300), 26, 3, PRIMAL, PRIMAL_EDGE, PRIMAL_PARTICLE, phase=math.pi / 2)
flat_springl((EX + 110, AY + 300), 28, 4, PRIMAL, PRIMAL_EDGE, PRIMAL_PARTICLE, phase=math.pi / 4)
flat_springl((EX + 195, AY + 300), 30, 6, DUAL, DUAL_EDGE, DUAL_PARTICLE, phase=0.0)
text(EX + 30, AY + 340, "triangle", 12, DIM_TEXT, "middle")
text(EX + 110, AY + 340, "quad (default)", 12, DIM_TEXT, "middle")
text(EX + 195, AY + 340, "dual n-gon", 12, DIM_TEXT, "middle")
paragraph(EX, AY + 364, ["Primal: an iso-surface face. Dual: one",
                         "vertex per face around its vertex (3–16)."], 12, DIM_TEXT, 16)

paragraph(EX, AY + 60, ["A 3D springl is a small polygon: a",
                                   "particle p with arms to its vertexes,",
                                   "a normal, and a capsule (the polygon",
                                   "thickened by a sphere). Particles and",
                                   "vertexes are advected by the flow;",
                                   "each springl then takes the rotation",
                                   "and uniform scale that best fit its",
                                   "advected vertexes, so it stays rigid,",
                                   "and its vertexes relax toward the",
                                   "nearest edges of other springls."])

# ---- (b)-(d) Primal and dual springls on an iso-surface patch -----------------------------------
# A non-uniform quad patch: a fan of polygons (two triangles, two quads, a pentagon) around a
# center vertex, each split into quads through its centroid and edge midpoints (one Catmull-Clark
# split without smoothing). Every face is a quad, but interior vertexes have valence 3 (triangle
# centroids), 4 or 5 (the center, the pentagon's centroid), so the dual springls are triangles,
# quads and pentagons.
CENTER = (2.3, 2.1)
RING_ANGLES = [-100, -55, -10, 35, 70, 115, 160, 200, 240]  # degrees; spokes at ring 0, 1, 3, 6, 7
RING_RADII = [2.3, 2.0, 2.4, 2.2, 2.0, 2.35, 2.1, 2.3, 2.15]
SPOKES = [0, 1, 3, 6, 7]


def height(x, y):
    return 0.4 * math.sin(0.9 * x + 0.2) * math.cos(0.7 * y - 0.5) + 0.1 * y


def point(x, y):
    return (x, y, height(x, y))


ring = [(CENTER[0] + r * math.cos(math.radians(a)), CENTER[1] + r * math.sin(math.radians(a)))
        for a, r in zip(RING_ANGLES, RING_RADII)]
base_points = [CENTER] + ring  # 0 is the center
base_faces = []
for k, s0 in enumerate(SPOKES):
    s1 = SPOKES[(k + 1) % len(SPOKES)]
    span = (s1 - s0) % len(ring)
    base_faces.append([0] + [1 + (s0 + t) % len(ring) for t in range(span + 1)])

# Split each base polygon into quads (all in the xy plane, counterclockwise; lifted by height()).
points2 = list(base_points)
midpoint_of = {}


def midpoint(a, b):
    key = (min(a, b), max(a, b))
    if key not in midpoint_of:
        midpoint_of[key] = len(points2)
        points2.append(tuple((x + y) / 2 for x, y in zip(points2[a], points2[b])))
    return midpoint_of[key]


faces = []
for f in base_faces:
    c = len(points2)
    points2.append(tuple(sum(points2[v][d] for v in f) / len(f) for d in range(2)))
    for k, v in enumerate(f):
        faces.append([v, midpoint(v, f[(k + 1) % len(f)]), c, midpoint(f[k - 1], v)])
verts = [point(x, y) for x, y in points2]
face_centroid = [centroid([verts[v] for v in f]) for f in faces]

# Dual springl of each interior vertex: the centroids of its faces, counterclockwise around it.
faces_of = {}
edge_count = {}
for n, f in enumerate(faces):
    for k, v in enumerate(f):
        faces_of.setdefault(v, []).append(n)
        e = (min(v, f[(k + 1) % 4]), max(v, f[(k + 1) % 4]))
        edge_count[e] = edge_count.get(e, 0) + 1
boundary = {v for e, n in edge_count.items() if n == 1 for v in e}
dual = []
for v, around in faces_of.items():
    if v in boundary:
        continue
    px, py = points2[v]
    around = sorted(around, key=lambda n: math.atan2(face_centroid[n][1] - py, face_centroid[n][0] - px))
    dual.append(([face_centroid[n] for n in around], verts[v]))
primal = [([verts[v] for v in f], c) for f, c in zip(faces, face_centroid)]
valence = {v: len(faces_of[v]) for v in faces_of}


def shrink(poly, center, s):
    return [lerp(center, v, s) for v in poly]


def scene(ox, oy, title, subtitle, show_primal, show_dual, s):
    emit(f'<rect x="{ox}" y="{oy}" width="300" height="470" rx="6" fill="{PANEL}"/>')
    text(ox + 14, oy + 26, title, 17, TEXT, weight="bold")
    paragraph(ox + 14, oy + 48, subtitle, 12, DIM_TEXT, 16)
    # Fit the patch into the panel below the subtitle.
    unit = View((0.0, 0.0), 1.0, azimuth=-20.0, elevation=60.0)
    xs, ys = zip(*(unit(x) for x in verts))
    box = (ox + 18, oy + 140, 264, 310)  # x, y, width, height
    scale = min(box[2] / (max(xs) - min(xs)), box[3] / (max(ys) - min(ys)))
    origin = (box[0] + (box[2] - scale * (max(xs) - min(xs))) / 2 - scale * min(xs),
              box[1] + (box[3] - scale * (max(ys) - min(ys))) / 2 - scale * min(ys))
    v = View(origin, scale, azimuth=-20.0, elevation=60.0)
    items = [(v.depth(c), "surface", [verts[x] for x in f], None) for f, c in zip(faces, face_centroid)]
    if show_primal:
        items += [(v.depth(c) - 0.01, "primal", shrink(poly, c, s), c) for poly, c in primal]
    if show_dual:
        items += [(v.depth(c) - 0.02, "dual", shrink(poly, c, s), c) for poly, c in dual]
    for _, kind, poly, c in sorted(items, key=lambda t: -t[0]):
        pts = [v(x) for x in poly]
        if kind == "surface":
            polygon(pts, SURFACE, SURFACE_EDGE, 1)
        elif kind == "primal":
            polygon(pts, PRIMAL, PRIMAL_EDGE, 1.2)
        else:
            polygon(pts, DUAL if not show_primal else "rgba(77,107,158,0.6)", DUAL_EDGE, 1.2)
    if show_primal:
        for _, c in primal:
            circle(v(c), 3.5, PRIMAL_PARTICLE)
    if show_dual:
        for _, c in dual:
            circle(v(c), 3.5, DUAL_PARTICLE)
    return v


BY = 440
tri_center = next(len(base_points) + k for k in range(len(points2) - len(base_points))
                  if valence.get(len(base_points) + k) == 3)
v = scene(20, BY, "(b) Primal springls", ["One springl per iso-surface face, its", "particle at the face centroid (drawn", "shrunk to show each springl)."],
          True, False, 0.84)
leader((40, BY + 112), v(face_centroid[2]), "primal particle")
leader((190, BY + 112), v(verts[1 + 2]), "iso-surface", "start")

v = scene(340, BY, "(c) Dual springls", ["One springl per iso-surface vertex: the",
                                         "dual-mesh face through the centroids", "of the faces around it (drawn shrunk)."],
          False, True, 0.84)
leader((360, BY + 112), v(verts[0]), "valence 5: a dual pentagon")
leader((360, BY + 455), v(verts[tri_center]), "valence 3: a dual triangle")

v = scene(660, BY, "(d) Doubleweave", ["Both constellations, staggered: dual", "particles sit where primal corners meet,",
                                       "dual corners reach the primal particles."],
          True, True, 1.0)
leader((680, BY + 112), v(verts[0]), "primal corners meet at a dual particle")

emit("</svg>")
path = Path(__file__).resolve().parent / "springls-anatomy-3d.svg"
path.write_text("\n".join(out) + "\n")
print(f"wrote {path}")
