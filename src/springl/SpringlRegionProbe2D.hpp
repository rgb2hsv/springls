#pragma once
#include <dweave/Manifold2D.hpp>
#include <spatial/SegmentTree2D.hpp>

#include <functional>
#include <utility>
#include <vector>

namespace springls
{
// Which region a point lies in, according to nearby labeled springls (the particle-sample
// correction of multi-object springls). Each region's springls are searched separately: at a
// boundary between two regions the springls of both sides coincide, and the nearest springl of
// any label would often be the other side's, whose normal points the wrong way.
class SpringlRegionProbe2D
{
public:
    // Springls with labels (springlLabels and springlParticleNormals parallel to the springls);
    // only springls for which include(n) holds, if given (e.g. one DoubleWeave constellation).
    explicit SpringlRegionProbe2D(const Manifold2D &manifold, const std::function<bool(size_t)> &include = nullptr);

    // True if q lies within `reach` of a springl, beside it (not past one of its ends, where the
    // side is ambiguous) and on its inner side (against its outward normal); then label and
    // distance are those of the nearest such springl (ties to the smaller label).
    bool inside(const Vec2 &q, float reach, int &label, float &distance) const;

private:
    const Manifold2D &mManifold;
    std::vector<std::pair<int, SegmentTree2D>> mTrees;  // one per label, ascending
};
}  // namespace springls
