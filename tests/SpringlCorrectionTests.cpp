// Label-aware particle-sample correction: band nodes beside a springl, on its inner side, take
// the springl's region.
#include "Check.hpp"

#include <levelset/MultiNarrowBand2D.hpp>
#include <scenarios/EnrightTest2D.hpp>
#include <spatial/SegmentTree2D.hpp>
#include <springl/MultiSpringLevelSet2D.hpp>
#include <springl/SpringlRegionProbe2D.hpp>

#include <cmath>
#include <cstdio>
#include <map>

using namespace springls;

namespace
{
// Two coincident springls on the segment (5, 5)-(6, 5), one per side of a shared boundary: label 1
// runs +x (normal +y, so its region lies below), label 2 runs -x (its region lies above).
Manifold2D SharedBoundary()
{
    Manifold2D m;
    m.springlParticles = {Vec2(5.5f, 5.0f), Vec2(5.5f, 5.0f)};
    m.springlVertexes = {Vec2(5, 5), Vec2(6, 5), Vec2(6, 5), Vec2(5, 5)};
    m.springlLabels = {1, 2};
    m.updateNormals();
    return m;
}

// Mean distance, over every step and springl, from a springl to the contour of its own region.
// (The largest distance is capped by contraction at about contractDistance either way.)
double MeanTracking(bool correction)
{
    EnrightConfig2D config;
    config.gridSize = 64;
    config.period = 1.0;
    MultiSpringLevelSet2D sim;
    sim.tuning.springlCorrection = correction;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    double sum = 0.0;
    size_t count = 0;
    while (sim.step()) {
        const Manifold2D &m = sim.getManifold();
        std::map<int, SegmentTree2D> own;
        for (size_t c = 0; c < m.vertexIndexes.size(); c++) {
            const Curve &curve = m.vertexIndexes[c];
            for (size_t k = 0; k + 1 < curve.size(); k++) own[m.curveLabels[c]].append(m.vertexLocations[curve[k]], m.vertexLocations[curve[k + 1]], 0);
        }
        for (auto &entry : own) entry.second.finish();
        for (size_t n = 0; n < m.springlCount(); n++) {
            const auto it = own.find(m.springlLabels[n]);
            if (it == own.end()) continue;
            sum += it->second.distance(m.springlParticles[n], 100.0f);
            count++;
        }
    }
    return (count > 0) ? sum / count : 0.0;
}
}  // namespace

TEST(springl_probe_picks_inner_region_at_shared_boundary)
{
    const Manifold2D m = SharedBoundary();
    const SpringlRegionProbe2D probe(m);
    int label = -1;
    float d = 0.0f;
    CHECK(probe.inside(Vec2(5.5f, 4.6f), 1.0f, label, d));
    CHECK(label == 1);
    CHECK_NEAR(d, 0.4f, 1E-5f);
    CHECK(probe.inside(Vec2(5.5f, 5.3f), 1.0f, label, d));
    CHECK(label == 2);
    CHECK_NEAR(d, 0.3f, 1E-5f);
}

TEST(springl_probe_ignores_points_past_ends_and_out_of_reach)
{
    const Manifold2D m = SharedBoundary();
    const SpringlRegionProbe2D probe(m);
    int label = -1;
    float d = 0.0f;
    CHECK(!probe.inside(Vec2(6.5f, 4.8f), 1.0f, label, d));  // past the end: side is ambiguous
    CHECK(!probe.inside(Vec2(5.5f, 3.5f), 1.0f, label, d));  // farther than the reach
}

TEST(multi_band_correct_labels_relabels_and_refreshes)
{
    constexpr int kSize = 32;
    ImageI labels(kSize, kSize);
    ImageF distance(kSize, kSize);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            labels(i, j) = (i < 16) ? 1 : 2;
            distance(i, j) = std::abs(i - 15.5f);
        }
    }
    MultiNarrowBand2D band;
    band.initialize(labels, distance);
    const int changed = band.correctLabels([](int i, int j, int &label, float &d) {
        if (i != 16 || j != 10) return false;
        label = 1;
        d = 0.2f;
        return true;
    });
    CHECK(changed == 1);
    CHECK(band.labels()(16, 10) == 1);
    CHECK(band.previousLabels()(16, 10) == 1);
    CHECK_NEAR(band.distance()(16, 10), 0.2f, 1E-6f);
    // The corrected pixel's right neighbor now has label 1 among its candidates.
    bool found = false;
    for (size_t n = 0; n < band.active().size(); n++) {
        if (band.active()[n] != Vec2i(17, 10)) continue;
        for (int c : band.candidates()[n]) found |= c == 1;
    }
    CHECK(found);
}

TEST(multi_springls_correction_tightens_tracking)
{
    const double with = MeanTracking(true);
    const double without = MeanTracking(false);
    std::printf("  mean springl distance to own contour: with correction %.4f px, without %.4f px\n", with, without);
    CHECK(with < without);
}
