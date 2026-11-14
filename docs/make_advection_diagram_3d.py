#!/usr/bin/env python3
"""Generates docs/springls-advection-3d.svg: one advection sub-step of a primal and a dual 3D springl
(SpringLevelSet3D::advect, then relaxSpringl).

Every stage is computed, not drawn by hand: a sample 3D flow (rotation, shear and bending), one RK4
step of the particle and of each vertex, the best-fit similarity transform of the previous planar
springl to the advected vertexes (Horn's quaternion method, as FitSimilarity3D), and one relaxation
step (arm springs, attraction to neighbor springl edges, rotation by the net moment).
Colors follow the enright3d panels. Run from anywhere: python3 docs/make_advection_diagram_3d.py
"""
import math
from pathlib import Path

BACKGROUND = "#202024"
PANEL = "#141418"
TEXT = "#e6e6e6"
DIM_TEXT = "#a8a8b0"
FLOW = "#4a5a4a"
TRAJECTORY = "#7fd17f"
PRIMAL = "rgba(158,82,77,0.80)"
PRIMAL_EDGE = "#d08078"
PRIMAL_PARTICLE = "#ff6a5c"
DUAL = "rgba(77,107,158,0.80)"
DUAL_EDGE = "#80a4e0"
DUAL_PARTICLE = "#6aa0ff"
VERTEX = "#ffcc00"
NORMAL = "#33cc33"
GHOST = "#8a8a94"
NEIGHBOR = "rgba(150,150,160,0.45)"
FONT = "Helvetica, Arial, sans-serif"

COLUMNS, ROWS = 4, 2
CELL_W, CELL_H = 250, 300
LEFT, TOP, GAP = 150, 112, 12
WIDTH = LEFT + COLUMNS * (CELL_W + GAP) + 8
HEIGHT = TOP + ROWS * (CELL_H + GAP) + 92
out = []


def emit(s):
    out.append(s)


def paint(attr, color):
    if color.startswith("rgba("):
        r, g, b, a = (float(v) for v in color[5:-1].split(","))
        return f'{attr}="#{int(r):02x}{int(g):02x}{int(b):02x}" {attr}-opacity="{a}"'
    return f'{attr}="{color}"'


def text(x, y, s, size=14, color=TEXT, anchor="start", weight="normal"):
    emit(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def line(a, b, color, width, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    emit(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" {paint("stroke", color)} stroke-width="{width}" stroke-linecap="round"{d}/>')


def circle(c, r, fill, stroke=None):
    s = f' {paint("stroke", stroke)} stroke-width="1.5"' if stroke else ""
    emit(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r}" {paint("fill", fill)}{s}/>')


def polyline(points, fill, stroke="none", width=1.0, closed=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    tag = "polygon" if closed else "polyline"
    emit(f'<{tag} points="{pts}" {paint("fill", fill)} {paint("stroke", stroke)} stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{d}/>')


def arrow(a, b, color, width=2.0, head=7):
    line(a, b, color, width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45))
    p2 = (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))
    polyline([p1, b, p2], "none", color, width, closed=False)


def spring(a, b, coils=4, amp=3.5):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    nx, ny = -dy / length, dx / length
    pts = [a]
    for k in range(1, coils * 2):
        t = k / (coils * 2)
        s = amp if k % 2 else -amp
        pts.append((a[0] + dx * t + nx * s, a[1] + dy * t + ny * s))
    pts.append(b)
    polyline(pts, "none", DIM_TEXT, 1.2, closed=False)


# ---- vectors ------------------------------------------------------------------------------------
def add(a, b): return tuple(x + y for x, y in zip(a, b))
def sub(a, b): return tuple(x - y for x, y in zip(a, b))
def mul(a, s): return tuple(x * s for x in a)
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def norm(a): return math.sqrt(dot(a, a))
def unit(a): return mul(a, 1.0 / max(norm(a), 1e-12))


def fan_normal(p, verts):
    n = (0.0, 0.0, 0.0)
    for k in range(len(verts)):
        n = add(n, cross(sub(verts[k], p), sub(verts[(k + 1) % len(verts)], p)))
    return unit(n)


def rotate(q, v):
    """Rotates v by unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    u = (x, y, z)
    t = mul(cross(u, v), 2.0)
    return add(add(v, mul(t, w)), cross(u, t))


def fit_similarity(arms, targets):
    """Rotation (quaternion) and uniform scale s minimizing sum |s R a - b|^2 (Horn 1987)."""
    S = [[sum(a[i] * b[j] for a, b in zip(arms, targets)) for j in range(3)] for i in range(3)]
    Sxx, Sxy, Sxz = S[0]
    Syx, Syy, Syz = S[1]
    Szx, Szy, Szz = S[2]
    N = [[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
         [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
         [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
         [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]]
    shift = sum(abs(N[i][j]) for i in range(4) for j in range(4)) + 1.0
    q = (1.0, 0.0, 0.0, 0.0)
    for _ in range(500):  # power iteration on N + shift I: its top eigenvector is the rotation
        q = tuple(sum(N[i][j] * q[j] for j in range(4)) + shift * q[i] for i in range(4))
        q = mul(q, 1.0 / norm(q))
    rotated = [rotate(q, a) for a in arms]
    s = sum(dot(r, b) for r, b in zip(rotated, targets)) / sum(dot(a, a) for a in arms)
    return q, s


# ---- the flow and one RK4 step ------------------------------------------------------------------
def velocity(p):
    """Rotation about a tilted axis, plus shear and an out-of-plane bend: divergence-free enough
    for a picture, and enough to make an advected polygon non-planar."""
    x, y, z = p
    return (-0.55 * y + 0.25 * z + 0.30 * x, 0.55 * x - 0.30 * y, 0.45 * x * x - 0.25 * y + 0.10)


DT = 1.5


def rk4(p):
    k1 = velocity(p)
    k2 = velocity(add(p, mul(k1, 0.5 * DT)))
    k3 = velocity(add(p, mul(k2, 0.5 * DT)))
    k4 = velocity(add(p, mul(k3, DT)))
    end = add(p, mul(add(add(k1, mul(k2, 2.0)), add(mul(k3, 2.0), k4)), DT / 6.0))
    stages = [p, add(p, mul(k1, 0.5 * DT)), add(p, mul(k2, 0.5 * DT)), add(p, mul(k3, DT))]
    return end, stages


def path(p, steps=24):
    pts = [p]
    for _ in range(steps):
        dt = DT / steps
        k1 = velocity(p)
        k2 = velocity(add(p, mul(k1, 0.5 * dt)))
        k3 = velocity(add(p, mul(k2, 0.5 * dt)))
        k4 = velocity(add(p, mul(k3, dt)))
        p = add(p, mul(add(add(k1, mul(k2, 2.0)), add(mul(k3, 2.0), k4)), dt / 6.0))
        pts.append(p)
    return pts


# ---- one relaxation step (relaxSpringl) ---------------------------------------------------------
def relax(particle, verts, targets, rest, sharpness=5.0, spring_constant=0.3, time_step=0.08):
    """Each vertex is pulled toward its neighbor-edge target (tangential motion turns the springl
    about its particle by the net moment) and its arm springs toward the rest length."""
    moment = (0.0, 0.0, 0.0)
    arms = []
    for v, t in zip(verts, targets):
        arm = sub(v, particle)
        length = norm(arm)
        tangent = unit(arm)
        motion = mul(sub(t, v), time_step * sharpness)
        spring_force = time_step * spring_constant * (rest - length)
        moment = add(moment, cross(motion, tangent))
        arms.append((length, tangent, motion, spring_force))
    angle = norm(moment)
    axis = unit(moment) if angle > 1e-9 else (0.0, 0.0, 1.0)
    half = -0.5 * angle
    q = (math.cos(half), *mul(axis, math.sin(half)))
    out = []
    for length, tangent, motion, spring_force in arms:
        radial = length + dot(motion, tangent) + spring_force
        out.append(add(particle, rotate(q, mul(tangent, radial))))
    return out


# ---- view ---------------------------------------------------------------------------------------
class View:
    def __init__(self, origin, scale, azimuth=-20.0, elevation=58.0):
        self.origin, self.scale = origin, scale
        self.ca, self.sa = math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))
        self.ce, self.se = math.cos(math.radians(elevation)), math.sin(math.radians(elevation))

    def __call__(self, p):
        x, y, z = p
        xr, yr = x * self.ca - y * self.sa, x * self.sa + y * self.ca
        return (self.origin[0] + self.scale * xr, self.origin[1] - self.scale * (yr * self.se + z * self.ce))


def draw_springl(view, particle, verts, fill, edge, particle_color, normal=True, dash=None, alpha_fill=True):
    pts = [view(v) for v in verts]
    if alpha_fill:
        for k in range(len(verts)):  # the fan, one triangle per arm pair
            polyline([view(particle), pts[k], pts[(k + 1) % len(pts)]], fill, "none")
    polyline(pts, "none", edge, 2.0, dash=dash)
    for k in range(len(verts)):
        line(view(particle), pts[k], edge, 1.0, dash=dash)
    if normal:
        arrow(view(particle), view(add(particle, mul(fan_normal(particle, verts), 0.9))), NORMAL, 2.0, 6)
    for p in pts:
        circle(p, 3.2, VERTEX)
    circle(view(particle), 4.5, particle_color)


def flow_hint(view):
    """Velocity arrows on a small grid around the springl."""
    for x in (-0.9, 0.0, 0.9):
        for y in (-0.9, 0.0, 0.9):
            if x == 0.0 and y == 0.0:
                continue
            p = (x, y, 0.0)
            arrow(view(p), view(add(p, mul(velocity(p), 0.35))), TRAJECTORY, 1.2, 5)


# ---- springls -----------------------------------------------------------------------------------
def regular(center, radii, phase):
    n = len(radii)
    return [add(center, (r * math.cos(phase + 2 * math.pi * k / n), r * math.sin(phase + 2 * math.pi * k / n), 0.0)) for k, r in enumerate(radii)]


rows = [
    ("Primal springl", ["an iso-surface face", "(4 vertexes)"], (0.0, 0.0, 0.0), regular((0.0, 0.0, 0.0), [0.6] * 4, math.radians(15)), PRIMAL, PRIMAL_EDGE, PRIMAL_PARTICLE),
    ("Dual springl", ["face centroids around", "an iso-surface vertex", "(3 to 16 vertexes)"], (0.0, 0.0, 0.0),
     regular((0.0, 0.0, 0.0), [0.58, 0.50, 0.61, 0.52, 0.60, 0.48], math.radians(5)), DUAL, DUAL_EDGE, DUAL_PARTICLE),
]
headers = [
    ("1. Springl at time t", ["planar: particle, arm vertexes,", "normal from the fan"]),
    ("2. RK4 per point", ["particle and every vertex move", "on their own; no longer planar"]),
    ("3. Best-fit similarity", ["previous shape, rotated and scaled", "about the moved particle: planar"]),
    ("4. Relaxation", ["arm springs, attraction to", "neighbor edges, net rotation"]),
]

emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}" role="img" aria-labelledby="title">')
emit('<title id="title">One advection sub-step of a primal and a dual 3D springl: RK4, best-fit similarity, relaxation</title>')
emit(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>')
text(20, 34, "3D springl advection: one sub-step", 22, TEXT, weight="bold")
text(WIDTH - 20, 34, "SpringLevelSet3D::advect, then relaxSpringl", 14, DIM_TEXT, "end")
for c, (title, lines) in enumerate(headers):
    x = LEFT + c * (CELL_W + GAP)
    text(x + 8, 68, title, 15, TEXT, weight="bold")
    for k, s in enumerate(lines):
        text(x + 8, 86 + 16 * k, s, 12, DIM_TEXT)

for r, (name, sub_name, particle, verts, fill, edge, pcolor) in enumerate(rows):
    y = TOP + r * (CELL_H + GAP)
    text(14, y + CELL_H / 2 - 14, name, 14, TEXT, weight="bold")
    for k, s_line in enumerate(sub_name):
        text(14, y + CELL_H / 2 + 4 + 15 * k, s_line, 11, DIM_TEXT)
    # The whole sub-step, computed once.
    moved_particle, stages = rk4(particle)
    moved = [rk4(v)[0] for v in verts]
    arms = [sub(v, particle) for v in verts]
    q, s = fit_similarity(arms, [sub(v, moved_particle) for v in moved])
    fitted = [add(moved_particle, mul(rotate(q, a), s)) for a in arms]
    # Neighbor springl edges beside each vertex: the attraction pulls the vertex toward the closest
    # point on them (here a short tangential offset, as for a neighbor rotated relative to this one).
    n_fit = fan_normal(moved_particle, fitted)
    targets = []
    for k, v in enumerate(fitted):
        arm = unit(sub(v, moved_particle))
        tangent = unit(cross(n_fit, arm))
        targets.append(add(v, add(mul(tangent, 0.16), mul(arm, -0.05))))
    rest = 0.95 * sum(norm(a) for a in arms) / len(arms) * s
    relaxed = relax(moved_particle, fitted, targets, rest)
    # Each stage's cell, framed on what it shows.
    frames = [verts + [particle], verts + moved + [particle, moved_particle], moved + fitted + [moved_particle], fitted + relaxed + targets]
    for c in range(COLUMNS):
        x = LEFT + c * (CELL_W + GAP)
        emit(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6" fill="{PANEL}"/>')
        emit(f'<clipPath id="clip{r}{c}"><rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6"/></clipPath>')
        emit(f'<g clip-path="url(#clip{r}{c})">')
        probe = View((0, 0), 1.0)
        xs = [probe(p)[0] for p in frames[c]]
        ys = [probe(p)[1] for p in frames[c]]
        scale = min((CELL_W - 70) / max(1e-6, max(xs) - min(xs)), (CELL_H - 90) / max(1e-6, max(ys) - min(ys)), 105)
        cx, cy = 0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys))
        view = View((x + CELL_W / 2 - scale * cx, y + CELL_H / 2 + 10 - scale * cy), scale)
        if c == 0:
            flow_hint(view)
            draw_springl(view, particle, verts, fill, edge, pcolor)
            text(x + 10, y + CELL_H - 14, "green: the flow's velocity", 11, DIM_TEXT)
        elif c == 1:
            draw_springl(view, particle, verts, fill, edge, pcolor, normal=False, dash="4,3", alpha_fill=False)
            # The advected springl, semi-transparent like the other stages (its fan is no longer flat).
            draw_springl(view, moved_particle, moved, fill, edge, pcolor, normal=False)
            for v in verts + [particle]:
                polyline([view(p) for p in path(v)], "none", TRAJECTORY, 1.4, closed=False)
            # RK4 stage points of the particle: x, x + dt/2 k1, x + dt/2 k2, x + dt k3.
            for st in stages:
                circle(view(st), 3.0, "none", TRAJECTORY)
            text(x + 10, y + CELL_H - 30, "green circles: the particle's RK4 stages", 11, DIM_TEXT)
            text(x + 10, y + CELL_H - 14, "dashed: at time t; solid: advected", 11, DIM_TEXT)
        elif c == 2:
            polyline([view(m) for m in moved], "none", GHOST, 1.4, dash="4,3")
            for m, f in zip(moved, fitted):
                line(view(m), view(f), GHOST, 1.0, dash="2,3")
                circle(view(m), 2.4, GHOST)
            draw_springl(view, moved_particle, fitted, fill, edge, pcolor)
            text(x + 10, y + CELL_H - 30, f"rotation R, scale s = {s:.2f}", 11, DIM_TEXT)
            text(x + 10, y + CELL_H - 14, "particle stays where RK4 put it", 11, DIM_TEXT)
        else:
            for t in targets:  # neighbor springl edges, the attraction's targets lie on them
                d = unit(sub(t, moved_particle))
                a, b = add(t, mul(cross(n_fit, d), 0.22)), add(t, mul(cross(n_fit, d), -0.22))
                line(view(a), view(b), NEIGHBOR, 3.0)
            polyline([view(f) for f in fitted], "none", GHOST, 1.4, dash="4,3")
            for f, t in zip(fitted, targets):
                spring(view(f), view(t))
            draw_springl(view, moved_particle, relaxed, fill, edge, pcolor)
            text(x + 10, y + CELL_H - 30, "dashed: before relaxing", 11, DIM_TEXT)
            text(x + 10, y + CELL_H - 14, "grey bars: neighbor springl edges", 11, DIM_TEXT)
        emit("</g>")

# Legend.
ly = HEIGHT - 30
items = [("particle", PRIMAL_PARTICLE), ("vertex", VERTEX), ("normal", NORMAL), ("RK4 trajectory", TRAJECTORY), ("before this stage", GHOST)]
x = LEFT
for label, color in items:
    if label == "normal":
        arrow((x, ly + 5), (x + 16, ly - 7), color, 2.0, 5)
    elif label in ("RK4 trajectory", "before this stage"):
        line((x, ly), (x + 18, ly), color, 2.0, dash="4,3" if color == GHOST else None)
    else:
        circle((x + 7, ly), 5 if label == "particle" else 4, color)
    text(x + 24, ly + 4.5, label, 12, TEXT)
    x += 24 + 7 * len(label) + 22

emit("</svg>")
path_out = Path(__file__).resolve().parent / "springls-advection-3d.svg"
path_out.write_text("\n".join(out) + "\n")
print(f"wrote {path_out} ({WIDTH} x {HEIGHT})")
