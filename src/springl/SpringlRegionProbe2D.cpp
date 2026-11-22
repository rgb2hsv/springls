#include <springl/SpringlRegionProbe2D.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace springls
{
SpringlRegionProbe2D::SpringlRegionProbe2D(const Manifold2D &manifold, const std::function<bool(size_t)> &include) : mManifold(manifold)
{
    std::map<int, SegmentTree2D> trees;
    for (size_t n = 0; n < manifold.springlParticles.size(); n++) {
        if (include && !include(n)) continue;
        trees[manifold.springlLabels[n]].append(manifold.springlVertexes[2 * n], manifold.springlVertexes[2 * n + 1], static_cast<int>(n));
    }
    for (auto &entry : trees) {
        entry.second.finish();
        mTrees.emplace_back(entry.first, std::move(entry.second));
    }
}

bool SpringlRegionProbe2D::inside(const Vec2 &q, float reach, int &label, float &distance) const
{
    constexpr float kEndTolerance = 1E-6f;
    float best = std::numeric_limits<float>::infinity();
    for (const auto &[treeLabel, tree] : mTrees) {
        const auto hit = tree.nearest(q, reach);
        if (!hit) continue;
        const size_t s = static_cast<size_t>(hit->id);
        if (distanceSqr(hit->point, mManifold.springlVertexes[2 * s]) < kEndTolerance ||
            distanceSqr(hit->point, mManifold.springlVertexes[2 * s + 1]) < kEndTolerance) {
            continue;
        }
        if (mManifold.springlParticleNormals[s].dot(q - hit->point) >= 0.0f) continue;
        const float d = std::sqrt(hit->distanceSqr);
        if (d < best) {
            best = d;
            label = treeLabel;
        }
    }
    if (!std::isfinite(best)) return false;
    distance = best;
    return true;
}
}  // namespace springls
