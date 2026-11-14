# Springls badge — design

## Intent
A badge for the repo that shows what springls are: a deforming surface carried by a
constellation of small, disconnected surface elements. It sits at the start of the
website's title block, "Springls (Spring Level Sets)", so the first thing a visitor sees
explains the method visually.

## Goals / non-goals
- Goal: the gaps between springls are visible, so the surface clearly reads as separate
  elements, not a drawn curve.
- Goal: it reads as *deforming*. The shape is a circle being stretched into the Enright
  swirl, with faint trails.
- Non-goal: a favicon or a tiny mark. This is a large badge only.
- Non-goal: animation (option C was not chosen).
- Anti-goal: anything that looks like a generic blob logo, or that clashes with the dark
  site theme.

## Approaches considered
- A. Small inline mark (about 48px) after the title. Too small to show the gaps.
- **B. Large badge (about 120px). Chosen, and placed to the LEFT of the title and subtitle.**
- C. Option B plus a CSS animation. Not chosen.

## Design
- One hand-written SVG, `docs/springls-badge.svg`, with a viewBox of 120×120 and no
  background, so it sits on the site's `--bg`.
- Content: 28 springls along a closed, swirled curve. Each springl has a gray arm, a
  particle in the middle and yellow vertices at the ends, matching the anatomy diagram.
  Particles alternate red `#e57373` and blue `#7fa7d9` (primal/dual, as in doubleweave),
  and each springl has a short green normal. The curve itself is never drawn, only the
  springls. Faint dashed arcs behind the springls show where they came from (motion trails).
- Placement: `.hero` gets a flex row with the badge on the left (120px, top-aligned with
  the h1) and the h1 and subtitle in a column on the right. Below about 640px wide, the
  badge stacks above the title at 96px.
- The README `# Springls` heading gets the same badge to its left, using an HTML
  `<img align="left">`. This is my call and easy to drop.

## Left to the builder
The exact swirl shape and springl spacing, tuned by eye in the browser.

## Revision (2026-11-14)
The Enright swirl read as a thin sliver. At the user's request, the shape is now the blob from
the "Semi-Lagrangian doubleweave: one sub-step" figure, r(θ) = 12 + 2.2 cos(3θ + 0.5). It sits
in front of a dashed outline of the undeformed circle it started as, which replaces the motion
trails. Untilted, no animation.
