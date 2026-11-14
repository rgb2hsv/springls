#include <springls/SpringlHashGrid3D.hpp>

#include <map>

namespace springls
{
namespace
{
// Level cell sizes are kFinestCell * 2^level: a springl goes to the finest level whose cell is at
// least its longest arm (uniform springls, with arms around half a voxel, land in 1-voxel cells).
constexpr float kFinestCell = 1.0f / 256.0f;
constexpr int kMaxLevel = 16;
}  // namespace

void SpringlHashGrid3D::Buckets::build(const Points3 &particles, const std::vector<uint32_t> &springls)
{
    std::vector<std::pair<uint64_t, uint32_t>> entries;
    entries.reserve(springls.size());
    for (uint32_t s : springls) {
        const Vec3 &p = particles[s];
        entries.emplace_back(Key(coord(p.x()), coord(p.y()), coord(p.z())), s);
    }
    std::sort(entries.begin(), entries.end());
    items.resize(entries.size());
    cells.clear();
    cells.reserve(entries.size());
    for (size_t e = 0; e < entries.size(); e++) {
        items[e] = entries[e].second;
        auto [it, inserted] = cells.try_emplace(entries[e].first, static_cast<uint32_t>(e), static_cast<uint32_t>(e + 1));
        if (!inserted) it->second.second = static_cast<uint32_t>(e + 1);
    }
}

void SpringlHashGrid3D::update(const Points3 &particles, const Points3 &vertexes, const std::vector<uint32_t> &start)
{
    mStart = start;
    mLevels.clear();
    const size_t N = particles.size();
    if (N == 0) return;
    // Level and reach of each springl.
    std::map<int, std::vector<uint32_t>> byLevel;  // finest first
    std::map<int, float> reachOf;
    for (size_t s = 0; s < N; s++) {
        float reachSqr = 0.0f;
        for (uint32_t v = start[s]; v < start[s + 1]; v++) reachSqr = std::max(reachSqr, distanceSqr(particles[s], vertexes[v]));
        const float reach = std::sqrt(reachSqr);
        int level = 0;
        while (level < kMaxLevel && std::ldexp(kFinestCell, level) < reach) level++;
        byLevel[level].push_back(static_cast<uint32_t>(s));
        float &r = reachOf[level];
        r = std::max(r, reach);
    }
    for (const auto &[level, springls] : byLevel) {
        Level out;
        out.reach = reachOf[level];
        out.fine.cellSize = std::ldexp(kFinestCell, level);
        out.coarse.cellSize = out.fine.cellSize * kTierRatio;
        out.fine.build(particles, springls);
        out.coarse.build(particles, springls);
        mLevels.push_back(std::move(out));
    }
}
}  // namespace springls
