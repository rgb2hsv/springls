#pragma once
#include <springls/SegmentTree2D.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

namespace springls
{
struct SignFieldParams {
    float capsuleCells = 1.5f;  // capsule radius r = max(capsuleCells * cellSize(p), kMinBandRadius * rasterSpacing, minBandPixels)
    // Absolute floor on the capsule radius (base pixels). Springls split after a refinement keep the
    // gaps they had before the split until fill closes them, and capsules of the new, smaller scale
    // do not overlap across them; without this floor the flood leaks through such a gap.
    float minBandPixels = 0.3f;
};

struct SignFieldInput {
    int width = 0, height = 0;                    // base-pixel domain [0,width-1] x [0,height-1]
    const Points2 *vertexes = nullptr;            // 2 per springl
    std::vector<uint32_t> springls;               // the springls that define this field
    std::function<float(const Vec2 &)> cellSize;  // local cell size h
    // Spacing of the sign raster (base pixels). Use the finest cell size in play: features thinner
    // than the raster resolves (about 2 * kMinBandRadius * rasterSpacing) are lost.
    float rasterSpacing = 1.0f;
};

// Level set reconstructed from springls (SemiLagrangianDoubleweaveLevelSet2D). As in the original
// spring level sets, the springls define only an unsigned distance; inside and outside come from
// topology, not from springl normals. On a raster of samples at (i, j) * rasterSpacing:
//
// 1. U is the min distance to the springl segments (the distance to the union of their capsules).
//    The capsules, the samples with U <= r, are walls.
// 2. The exterior floods in from the raster's border around the walls (4-connected); samples it
//    does not reach are inside (so an outside pocket sealed off by walls is filled).
// 3. The sign propagates upwind into the walls: wall samples in order of decreasing U, each taking
//    the sign of its neighbors with larger U weighted by how much larger, so the outside and
//    inside fronts meet at U = 0, on the springls. A sample with no upwind neighbor (a local
//    maximum of U inside a wall: the middle of a feature thinner than two capsules) is inside, so
//    thin filaments survive; the price is that fronts closer than two capsules fuse (a union).
//    On the Enright test it beat reading the previous field's sign there (64^2, T 8, depth 4:
//    -0.23% area error against -1.37%) and defaulting to outside (128^2, T 8, depth 2: -0.17%
//    against -0.67%).
// 4. phi = sign * min(U, clamp).
//
// With no springls at all there is no surface: outside everywhere. The input's cellSize callback
// must be thread-safe (build() calls it from OpenMP threads), and the vertex array must outlive
// the field.
class SpringlSignField2D
{
public:
    // Smallest capsule radius, in raster samples (times rasterSpacing): a wall thinner than half a
    // sample spacing can pass between samples and let the flood leak through.
    static constexpr float kMinBandRadius = 0.75f;

    void build(const SignFieldInput &input, const SignFieldParams &params);
    float distance(const Vec2 &p, float maxDistance) const;  // U(p): min distance to the springl segments
    float sign(const Vec2 &p) const;                         // +1 outside, -1 inside (the nearest sample's)
    float phi(const Vec2 &p, float clamp) const { return sign(p) * std::min(distance(p, clamp), clamp); }

    // Diagnostics.
    size_t wallSampleCount() const { return mWallCount; }

private:
    int pixelOf(const Vec2 &p) const;
    float bandRadius(const Vec2 &p) const;  // r at p

    SignFieldInput mInput;
    SignFieldParams mParams;
    SegmentTree2D mTree;
    std::vector<int8_t> mPixelSign;  // per raster sample
    size_t mWallCount = 0;
    float mSpacing = 1.0f;
    int mRasterWidth = 0, mRasterHeight = 0;
};
}  // namespace springls
