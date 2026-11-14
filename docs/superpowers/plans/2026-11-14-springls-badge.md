# Springls Badge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 120px SVG badge of a deforming springl constellation, shown to the left of the
site's hero title and the README heading.

**Architecture:** A generator script `docs/make_badge.py`, following the existing
`docs/make_*_diagram.py` convention: stdlib only, string-built SVG, run from anywhere. It
writes `docs/springls-badge.svg`. The site places the badge with a flex row in `.hero`; the
README uses `<img align="left">`.

**Tech Stack:** Python 3 (stdlib), SVG, HTML/CSS.

**Spec:** `docs/superpowers/specs/2026-11-14-badge-design.md`

## Global Constraints
- viewBox `0 0 120 120`, no background rect.
- Exactly 28 springls; the curve is never drawn.
- Colors: particles alternate `#e57373` / `#7fa7d9`; vertices yellow (`#f2c94c`); arm gray
  (`#9a9a9a`); normal green (`#7bd88f`); trails `#8b919e` dashed at low opacity.
- No animation.
- Site: badge 120px on the left at ≥640px wide; stacked above at 96px below 640px.

## Review Focus
1. Narrow screens (≤640px): the badge must not squeeze the title. It stacks above it. Checked
   in Task 2 at a 375px viewport.
2. Gaps between springls must stay visible at 120px. Task 1 asserts that the arm length is
   less than the spacing between centers.
3. Normals point outward from the curve, not inward. Task 1 asserts this with a centroid check.
4. The SVG must render as an `<img>` on GitHub (no external refs or scripts). Task 1 asserts
   this.
5. The hero heading text and anchor links still work. Task 2 checks the h1 text is unchanged.

---

### Task 1: Badge generator and SVG

**Files:**
- Create: `docs/make_badge.py`
- Create: `docs/springls-badge.svg` (generated)

**Interfaces:**
- Produces: `docs/springls-badge.svg`, 120×120, used by Task 2.
- `curve(t: float) -> tuple[float, float]`: closed swirl, t in [0, 1). It is a circle
  deformed by a single-vortex shear (Enright-like), centered at (60, 60), radius about 38.
- `springls(n: int = 28) -> list[tuple[p, a, b, nrm]]`: particle, arm ends and unit
  outward normal, placed at equal arc length.

- [ ] **Step 1:** Write `docs/make_badge.py`, with self-checks run in `__main__` before
  writing:
  `assert len(S) == 28`; for each springl, `dist(a, b) < 0.8 * spacing` (gaps visible);
  `dot(nrm, p - centroid) > 0` (outward); the output contains no `href`, `<script` or
  `<image`.
- [ ] **Step 2:** Run `python3 docs/make_badge.py`. Expected: it prints
  `wrote docs/springls-badge.svg` with no assertion errors.
- [ ] **Step 3:** Open the SVG in the browser pane at 120px, and tune the swirl shape and
  trail length by eye.
- [ ] **Step 4:** Commit with
  `git add docs/make_badge.py docs/springls-badge.svg && git commit -m "Add springls badge"`.

### Task 2: Place the badge on the site and README

**Files:**
- Modify: `docs/index.html:80-82` (wrap the h1 and subtitle in `<div class="hero-text">`
  and add `<img class="badge" src="springls-badge.svg" alt="Springls badge" width="120" height="120">` before it, inside `<div class="hero-head">`)
- Modify: `docs/style.css` (after `.hero h1`): `.hero-head { display:flex; gap:24px;
  align-items:flex-start; }`, `.badge { flex:none; }`, and
  `@media (max-width:640px) { .hero-head { flex-direction:column; gap:12px } .badge { width:96px; height:96px } }`
- Modify: `README.md:1` (`<img src="docs/springls-badge.svg" align="left" width="120"
  alt="">` above the heading, followed by `<br clear="left">` after the first paragraph)

- [ ] **Step 1:** Make the edits.
- [ ] **Step 2:** Serve `docs/` (launch config `docs`). At desktop width, the badge's left
  edge is at x < h1's left edge and the tops are aligned within 8px. At 375px, the badge is
  above the h1 and is 96px wide. `h1.textContent` still reads
  `Springls (Spring Level Sets)`.
- [ ] **Step 3:** Commit with
  `git add docs/index.html docs/style.css README.md && git commit -m "Show badge in site header and README"`.
