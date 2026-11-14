#pragma once
// Shared types and helpers for the springls library: Eigen vector aliases, small 2D vector
// helpers, and the optional OpenMP pragma wrapper.
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace springls
{
using Vec2 = Eigen::Vector2f;
using Vec2d = Eigen::Vector2d;
using Vec2i = Eigen::Vector2i;
using Points2 = std::vector<Vec2>;
// Iso-contour curve as indexes into a vertex array. Closed curves repeat their first index at the end.
using Curve = std::vector<uint32_t>;
using Curves = std::vector<Curve>;

constexpr float kPi = 3.14159265358979323846f;

inline float cross2(const Vec2 &a, const Vec2 &b) { return a.x() * b.y() - a.y() * b.x(); }
inline float distance(const Vec2 &a, const Vec2 &b) { return (a - b).norm(); }
inline float distanceSqr(const Vec2 &a, const Vec2 &b) { return (a - b).squaredNorm(); }
// Unit vector along v, or zero when v is (nearly) zero.
inline Vec2 normalizedOrZero(const Vec2 &v)
{
    const float n = v.norm();
    return (n > 1E-12f) ? Vec2(v / n) : Vec2::Zero();
}
inline float signOf(float a) { return (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f); }
inline Vec2 cwiseClamp(const Vec2 &p, const Vec2 &lo, const Vec2 &hi) { return p.cwiseMax(lo).cwiseMin(hi); }

// Squared distance from p to segment [a, b], and the closest point on it.
inline float SegmentDistanceSqr(const Vec2 &a, const Vec2 &b, const Vec2 &p, Vec2 *closest = nullptr)
{
    const Vec2 ab = b - a;
    const float len2 = ab.squaredNorm();
    const float t = (len2 > 0.0f) ? std::clamp((p - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
    const Vec2 q = a + t * ab;
    if (closest) *closest = q;
    return (p - q).squaredNorm();
}

// Optional OpenMP: SPRINGLS_OMP(omp parallel for if(n >= kParallelMinLightItems)) expands to the
// pragma when built with SPRINGLS_OPENMP, and to nothing otherwise.
#define SPRINGLS_STRINGIFY(...) #__VA_ARGS__
#ifdef SPRINGLS_OPENMP
#define SPRINGLS_OMP(...) _Pragma(SPRINGLS_STRINGIFY(__VA_ARGS__))
#else
#define SPRINGLS_OMP(...)
#endif
// Minimum loop sizes for a parallel region; below these the fork/join cost outweighs the work.
constexpr int kParallelMinLightItems = 16384;  // a few arithmetic operations per item
constexpr int kParallelMinHeavyItems = 256;    // a spatial query or full stencil per item
constexpr int kParallelMinBlocks = 16;         // adaptive-grid blocks (81 corners each)

// Number of threads OpenMP will use (1 without OpenMP).
int MaxThreads();
// Sets the OpenMP thread count (no-op without OpenMP). n <= 0 leaves the default.
void SetThreads(int n);
}  // namespace springls
