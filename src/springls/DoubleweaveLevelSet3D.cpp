#include <springls/DoubleweaveLevelSet3D.hpp>

#include <cassert>
#include <unordered_map>

namespace springls
{
void ForEachDualSpringl(const SurfaceMesh3D &mesh, const std::function<void(const Vec3 &, std::span<const Vec3>, uint32_t)> &dual)
{
    const int K = mesh.elementSize();
    const size_t F = mesh.faceCount();
    const Points3 &v = mesh.vertexes;
    // Face of every directed edge a -> b (-2 if more than one), and face centroids.
    Points3 centroids(F);
    std::unordered_map<uint64_t, int> faceOfEdge;
    faceOfEdge.reserve(F * K);
    std::vector<std::vector<std::pair<uint32_t, int>>> corners(v.size());  // (face, corner) per vertex
    for (size_t f = 0; f < F; f++) {
        const uint32_t *idx = mesh.face(f);
        Vec3 c = Vec3::Zero();
        for (int k = 0; k < K; k++) {
            c += v[idx[k]];
            const uint64_t key = (static_cast<uint64_t>(idx[k]) << 32) | idx[(k + 1) % K];
            auto [it, inserted] = faceOfEdge.try_emplace(key, static_cast<int>(f));
            if (!inserted) it->second = -2;
            corners[idx[k]].emplace_back(static_cast<uint32_t>(f), k);
        }
        centroids[f] = c / static_cast<float>(K);
    }
    std::vector<Vec3> cell;
    for (uint32_t vertex = 0; vertex < v.size(); vertex++) {
        const auto &around = corners[vertex];
        if (around.size() < 3) continue;
        // Walk the faces around the vertex counterclockwise: from face f, the next one is across
        // f's edge into the vertex, i.e. the face that leaves the vertex toward f's previous corner.
        cell.clear();
        uint32_t f = around[0].first;
        int k = around[0].second;
        bool closed = false;
        for (size_t step = 0; step <= around.size(); step++) {
            cell.push_back(centroids[f]);
            const uint32_t prev = mesh.face(f)[(k + K - 1) % K];
            const auto it = faceOfEdge.find((static_cast<uint64_t>(vertex) << 32) | prev);
            if (it == faceOfEdge.end() || it->second < 0) break;  // boundary or non-manifold
            f = static_cast<uint32_t>(it->second);
            const uint32_t *idx = mesh.face(f);
            k = 0;
            while (k < K && idx[k] != vertex) k++;
            if (f == around[0].first) {
                closed = true;
                break;
            }
        }
        if (!closed || cell.size() != around.size() || cell.size() > static_cast<size_t>(kMaxSpringlSize)) continue;
        dual(v[vertex], std::span<const Vec3>(cell.data(), cell.size()), vertex);
    }
}

DoubleweaveLevelSet3D::DoubleweaveLevelSet3D(ElementType elementType) : SpringLevelSet3D(elementType) { mName = "Doubleweave 3D"; }

size_t DoubleweaveLevelSet3D::countSpringls(Constellation constellation) const
{
    return static_cast<size_t>(std::count(mManifold.springlConstellations.begin(), mManifold.springlConstellations.end(), static_cast<uint8_t>(constellation)));
}

size_t DoubleweaveLevelSet3D::appendSpringl(const Vec3 &particle, std::span<const Vec3> vertexes, Constellation constellation)
{
    const size_t index = SpringLevelSet3D::appendSpringl(particle, vertexes);
    mManifold.springlConstellations.push_back(static_cast<uint8_t>(constellation));
    return index;
}

void DoubleweaveLevelSet3D::seedSpringls()
{
    const SurfaceMesh3D &iso = mManifold.isoSurface;
    const int K = elementSize();
    std::array<Vec3, kMaxElementSize> verts;
    for (size_t f = 0; f < iso.faceCount(); f++) {
        for (int k = 0; k < K; k++) verts[k] = iso.vertexes[iso.face(f)[k]];
        appendSpringl(Centroid(verts.data(), K), std::span<const Vec3>(verts.data(), K), Constellation::Primal);
    }
    ForEachDualSpringl(iso, [&](const Vec3 &particle, std::span<const Vec3> v, uint32_t) { appendSpringl(particle, v, Constellation::Dual); });
}

void DoubleweaveLevelSet3D::clearSpringls()
{
    SpringLevelSet3D::clearSpringls();
    mManifold.springlConstellations.clear();
}

void DoubleweaveLevelSet3D::compactSpringls(const std::vector<uint8_t> &keep)
{
    SpringLevelSet3D::compactSpringls(keep);
    size_t out = 0;
    for (size_t i = 0; i < keep.size(); i++) {
        if (keep[i]) mManifold.springlConstellations[out++] = mManifold.springlConstellations[i];
    }
    mManifold.springlConstellations.resize(out);
}

void DoubleweaveLevelSet3D::snapshotForTracking()
{
    SpringLevelSet3D::snapshotForTracking();
    mOldConstellations = mManifold.springlConstellations;
}

bool DoubleweaveLevelSet3D::canTrackFrom(size_t newIndex, size_t oldIndex) const
{
    return newIndex < mManifold.springlConstellations.size() && oldIndex < mOldConstellations.size() &&
           mManifold.springlConstellations[newIndex] == mOldConstellations[oldIndex];
}

bool DoubleweaveLevelSet3D::refinesContourWith(size_t particle) const
{
    return mActiveConstellation < 0 ||
           (particle < mManifold.springlConstellations.size() && mManifold.springlConstellations[particle] == static_cast<uint8_t>(mActiveConstellation));
}

bool DoubleweaveLevelSet3D::init()
{
    for (ConstellationLevelSet &state : mConstellations) state = ConstellationLevelSet();
    mActiveConstellation = -1;
    if (!SpringLevelSet3D::init()) return false;
    ensureConstellationLevelSets();
    evolveConstellationLevelSets();
    return true;
}

void DoubleweaveLevelSet3D::swapConstellation(size_t w)
{
    ConstellationLevelSet &state = mConstellations[w];
    std::swap(mBand, state.band);
    std::swap(mUnsignedLevelSet, state.unsignedLevelSet);
    std::swap(mManifold.isoSurface, state.isoSurface);
}

void DoubleweaveLevelSet3D::ensureConstellationLevelSets()
{
    for (ConstellationLevelSet &state : mConstellations) {
        if (state.valid) continue;
        state.band = mBand;
        state.unsignedLevelSet = mUnsignedLevelSet;
        state.isoSurface = mManifold.isoSurface;
        state.valid = true;
    }
}

void DoubleweaveLevelSet3D::evolveConstellationLevelSets()
{
    ensureConstellationLevelSets();
    for (size_t w = 0; w < kConstellationCount; w++) {
        swapConstellation(w);
        updateUnsignedLevelSet(tuning.unsignedMaxDistance, [this, w](size_t n) { return mManifold.springlConstellations[n] == w; });
        for (int i = 0; i < tuning.signedEvolveIterations; i++) updateSignedLevelSet();
        mActiveConstellation = static_cast<int>(w);
        solveContour();
        mActiveConstellation = -1;
        swapConstellation(w);
    }
}

int DoubleweaveLevelSet3D::contract()
{
    ensureConstellationLevelSets();
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    int contracted = 0;
    for (size_t w = 0; w < kConstellationCount; w++) {
        swapConstellation(w);
        contracted += markContractions(keep, [this, w](size_t n) { return mManifold.springlConstellations[n] == w; });
        swapConstellation(w);
    }
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int DoubleweaveLevelSet3D::fill()
{
    ensureConstellationLevelSets();
    // Gap tests per constellation, against the springls that exist before this pass: every gap is
    // found before any springl is appended, so the arrays can be read in place.
    refreshSpringlGrid();
    const int K = elementSize();
    const std::vector<uint8_t> &constellations = mManifold.springlConstellations;
    struct Candidate {
        Vec3 particle;
        std::array<Vec3, kMaxSpringlSize> vertexes;
        int size = 0;
    };
    std::array<std::vector<Candidate>, kConstellationCount> gaps;
    for (size_t w = 0; w < kConstellationCount; w++) {
        // Each constellation fills from its own iso-surface, exactly as single springls would.
        const SurfaceMesh3D &iso = mConstellations[w].isoSurface;
        std::vector<Candidate> candidates;
        if (w == static_cast<size_t>(Constellation::Primal)) {
            candidates.resize(iso.faceCount());
            for (size_t f = 0; f < iso.faceCount(); f++) {
                for (int k = 0; k < K; k++) candidates[f].vertexes[k] = iso.vertexes[iso.face(f)[k]];
                candidates[f].size = K;
                candidates[f].particle = Centroid(candidates[f].vertexes.data(), K);
            }
        } else {
            ForEachDualSpringl(iso, [&](const Vec3 &particle, std::span<const Vec3> v, uint32_t) {
                Candidate c;
                c.particle = particle;
                std::copy(v.begin(), v.end(), c.vertexes.begin());
                c.size = static_cast<int>(v.size());
                candidates.push_back(c);
            });
        }
        const int C = static_cast<int>(candidates.size());
        const uint8_t tag = static_cast<uint8_t>(w);
        std::vector<uint8_t> isGap(C, 0);
        SPRINGLS_OMP(omp parallel for if (C >= kParallelMinHeavyItems))
        for (int c = 0; c < C; c++) {
            isGap[c] = !mSpringlGrid.anyElementWithin(mManifold.springlParticles, mManifold.springlVertexes, candidates[c].particle, tuning.fillDistance,
                                                      [&](size_t n) { return constellations[n] == tag; });
        }
        for (int c = 0; c < C; c++) {
            if (isGap[c]) gaps[w].push_back(candidates[c]);
        }
    }
    int filled = 0;
    for (size_t w = 0; w < kConstellationCount; w++) {
        for (const Candidate &c : gaps[w]) {
            appendSpringl(c.particle, std::span<const Vec3>(c.vertexes.data(), c.size), static_cast<Constellation>(w));
            filled++;
        }
    }
    return filled;
}
}  // namespace springls
