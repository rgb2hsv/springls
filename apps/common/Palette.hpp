#pragma once
// The toybox's one color palette: every window (2D Enright, 3D Enright, mesh, sphere) draws its elements from
// these, so an element has the same color in every render. Values are 0..1 RGB; alpha, where a window blends,
// is the window's choice.
//
// Three tiers, so the components separate at a glance and still sit together:
// - a neutral charcoal base (window chrome; a dark matcap outside the regions);
// - recessive structure in muted cool tones (refinement levels, distance isolines), which stays behind;
// - high-contrast accents for the interface and springl elements, each in its own hue family (cyan contour,
//   slate springls, amber vertexes, mint normals, crimson particles, dark fuchsia arrows, ivory reference).
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace toybox::palette
{
struct Color {
    float r = 0, g = 0, b = 0;
    constexpr std::array<float, 3> rgb() const { return {r, g, b}; }
    constexpr Color scaled(float s) const { return {r * s, g * s, b * s}; }
};
constexpr Color Rgb8(int r, int g, int b) { return {r / 255.0f, g / 255.0f, b / 255.0f}; }
constexpr Color Hex(unsigned v) { return Rgb8(static_cast<int>((v >> 16) & 0xff), static_cast<int>((v >> 8) & 0xff), static_cast<int>(v & 0xff)); }

// Base: window chrome.
inline constexpr Color kBackground = Hex(0x1E1F24);
inline constexpr Color kPanel = Hex(0x15161A);
inline constexpr Color kText = Hex(0xE6E6E6);
inline constexpr Color kStatus = Hex(0xF5C542);
inline constexpr Color kTextShadow = Hex(0x0A0A0A);

// The interface: the level set's zero contour (2D, the sphere), and the matcap tint of a 3D iso-surface.
inline constexpr Color kContour = Hex(0x3FC1C9);     // bright cyan
inline constexpr Color kIsoSurface = Hex(0xEDEAE2);  // warm near white
// The level set's distance isolines (recessive).
inline constexpr Color kIsoline = Hex(0x8FA6D8);  // light soft blue
// The exact (Lagrangian) reference curve.
inline constexpr Color kReference = Hex(0xF2EBD3);  // ivory
// Velocity arrows.
inline constexpr Color kArrow = Hex(0x9C1B74);  // dark fuchsia

// Springl elements: primal and dual springls, their particles, arm vertexes, normals, and capsules.
inline constexpr Color kPrimalSpringl = Hex(0x8A93A6);   // slate
inline constexpr Color kPrimalParticle = Hex(0xD23A48);  // crimson
inline constexpr Color kDualSpringl = Hex(0xF08A24);     // orange
inline constexpr Color kDualParticle = Hex(0x2F7FD8);    // azure
inline constexpr Color kVertex = Hex(0xF5C542);          // amber
inline constexpr Color kNormal = Hex(0x5CD69A);          // mint
inline constexpr Color kPrimalCapsule = Hex(0xE05A8A);   // rose
inline constexpr Color kDualCapsule = Hex(0x4AA3E8);     // sky

// A single object (the sphere scenario's puzzle piece or star): soft warm white, so the springl elements and
// the contour read clearly on it and it stands apart from the dark background.
inline constexpr Color kObject = Hex(0xE8E4DA);

// The input mesh, where a window overlays it.
inline constexpr Color kInputMesh = Hex(0x8DBF7A);  // sage

// Regions of multi-object methods, by label (1 first; label 0, the background, is untinted): muted, evenly
// spaced hues, coral first.
inline constexpr std::array<Color, 6> kRegions = {Hex(0xD9795F), Hex(0x6FA8DC), Hex(0x8DBF7A), Hex(0xE3B65C), Hex(0xA88BD6), Hex(0x5FBFB0)};
constexpr Color Region(int label) { return kRegions[static_cast<size_t>(std::max(label, 1) - 1) % kRegions.size()]; }

// Refinement depth: adaptive grid cells (2D) and subdivision levels of a surface mesh (the sphere's wireframe),
// from the base (0) to the finest; deeper ones take the last. A muted cool ramp (slate, steel blue, indigo,
// violet, orchid), ordered by hue and clear of the accents' hue families.
inline constexpr std::array<Color, 5> kLevels = {Hex(0x5B6073), Hex(0x3E6F94), Hex(0x5560AA), Hex(0x7D5AA6), Hex(0xA65A9A)};
constexpr Color Level(int level) { return kLevels[static_cast<size_t>(std::clamp(level, 0, static_cast<int>(kLevels.size()) - 1))]; }

// Color of meshlet LOD level `level` of `levelCount`, as meshlet-viewer's "Meshlet (LOD gradient)" mode (its
// lod_palette.cpp): designed in CIELAB (D65), lightness falls in equal steps from the finest level (L* 92) to
// the coarsest (L* 62), hue turns 300 degrees from yellow round to red, and chroma is the largest (up to 55)
// that stays inside sRGB. Linear RGB.
inline Color MeshletLevelColor(int level, int levelCount)
{
    constexpr float kPiF = std::numbers::pi_v<float>;
    struct Rgb {
        float r, g, b;
    };
    const auto labToLinear = [](float L, float a, float b) {
        const auto finv = [](float t) {
            constexpr float d = 6.0f / 29.0f;
            return t > d ? t * t * t : 3.0f * d * d * (t - 4.0f / 29.0f);
        };
        const float fy = (L + 16.0f) / 116.0f;
        const float X = 0.95047f * finv(fy + a / 500.0f), Y = finv(fy), Z = 1.08883f * finv(fy - b / 200.0f);
        return Rgb{3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z, -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z,
                   0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z};
    };
    const auto inGamut = [](const Rgb &c) {
        for (float v : {c.r, c.g, c.b}) {
            if (v < -1e-4f || v > 1.0f + 1e-4f) return false;
        }
        return true;
    };
    const float t = levelCount > 1 ? static_cast<float>(std::min(level, levelCount - 1)) / static_cast<float>(levelCount - 1) : 0.0f;
    const float L = 92.0f + (62.0f - 92.0f) * t;
    const float hue = (100.0f + 300.0f * t) * kPiF / 180.0f;
    float lo = 0.0f, hi = 55.0f;
    if (!inGamut(labToLinear(L, hi * std::cos(hue), hi * std::sin(hue)))) {
        for (int i = 0; i < 24; i++) {
            const float mid = 0.5f * (lo + hi);
            (inGamut(labToLinear(L, mid * std::cos(hue), mid * std::sin(hue))) ? lo : hi) = mid;
        }
    } else {
        lo = hi;
    }
    const Rgb c = labToLinear(L, lo * std::cos(hue), lo * std::sin(hue));
    return {std::clamp(c.r, 0.0f, 1.0f), std::clamp(c.g, 0.0f, 1.0f), std::clamp(c.b, 0.0f, 1.0f)};
}
}  // namespace toybox::palette
