// AdaptiveMultiSpringLevelSet2D: multi-object springls on the labeled block-adaptive quadtree.
#include "Check.hpp"

#include <scenarios/EnrightTest2D.hpp>
#include <springl/AdaptiveMultiSpringLevelSet2D.hpp>

#include <cmath>
#include <cstdio>

using namespace springls;

namespace
{
EnrightConfig2D Config(int grid)
{
    EnrightConfig2D config;
    config.gridSize = grid;
    config.period = 1.0;
    return config;
}

struct SplitProbe : AdaptiveMultiSpringLevelSet2D {
    using AdaptiveMultiSpringLevelSet2D::splitSpringl;
};
}  // namespace

TEST(adaptive_multi_seeds_labeled_springls)
{
    const EnrightConfig2D config = Config(32);
    auto sim = MakeSimulation(Method::AdaptiveMultiSpringls, 2);
    CHECK(sim != nullptr);
    if (!sim) return;
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    const auto *adaptive = dynamic_cast<const AdaptiveMultiSpringLevelSet2D *>(sim.get());
    CHECK(adaptive != nullptr);
    const Manifold2D &m = sim->getManifold();
    CHECK(m.springlCount() > 0);
    CHECK(m.springlLabels.size() == m.springlCount());
    int badLabel = 0, coarse = 0;
    for (size_t n = 0; n < m.springlCount(); n++) {
        badLabel += (m.springlLabels[n] < 1 || m.springlLabels[n] > kEnrightRegionCount) ? 1 : 0;
        const AdaptiveGrid2D &grid = adaptive->getGrid();
        coarse += (grid.blocks()[grid.findLeaf(m.springlParticles[n])].key.depth != 2) ? 1 : 0;
    }
    CHECK(badLabel == 0);
    CHECK(coarse == 0);
}

TEST(adaptive_multi_no_vacuum)
{
    const EnrightConfig2D config = Config(32);
    auto sim = MakeSimulation(Method::AdaptiveMultiSpringls, 2);
    if (!sim) return;
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    for (int k = 0; k < 20; k++) sim->step();
    const AdaptiveGrid2D &grid = dynamic_cast<const AdaptiveMultiSpringLevelSet2D &>(*sim).getGrid();
    int bad = 0;
    for (int index : grid.leaves()) {
        const AdaptiveGrid2D::Block &block = grid.blocks()[index];
        for (int node = 0; node < AdaptiveGrid2D::kNodeCount; node++) {
            const int l = block.label(node);
            bad += (l < 0 || l > kEnrightRegionCount) ? 1 : 0;
        }
    }
    CHECK(bad == 0);
}

TEST(adaptive_multi_enright_headless)
{
    const EnrightConfig2D config = Config(32);
    auto sim = MakeSimulation(Method::AdaptiveMultiSpringls, 2);
    if (!sim) return;
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    int steps = 0;
    while (sim->step() && steps < 100000) steps++;
    const EnrightMetrics2D m = MeasureEnright(*sim, config);
    CHECK_NEAR(m.time, 1.0, 1E-9);
    CHECK(m.labelRelativeAreaErrors.size() == static_cast<size_t>(kEnrightRegionCount));
    for (size_t l = 0; l < m.labelRelativeAreaErrors.size(); l++) {
        std::printf("  adaptive multi label %zu area error %.4f\n", l + 1, m.labelRelativeAreaErrors[l]);
        CHECK(std::abs(m.labelRelativeAreaErrors[l]) < 0.06);  // first run: -0.036 -0.037 -0.009 -0.011
    }
}

// At depth 5 (32 base cells, 1024 equivalent) a sub-step of a third of a base pixel moved springls
// about 10 finest cells, further than the level set can follow toward them in one evolution: the
// contour fell behind and fill re-seeded springls along it (2,413 -> 8,725 in three steps).
TEST(adaptive_multi_depth5_tracks_springls)
{
    EnrightConfig2D config;
    config.gridSize = 32;
    config.period = 8.0;
    auto sim = MakeSimulation(Method::AdaptiveMultiSpringls, 5);
    if (!sim) return;
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    const size_t initial = sim->getManifold().springlCount();
    for (int k = 0; k < 3; k++) sim->step();
    const auto &adaptive = dynamic_cast<const AdaptiveMultiSpringLevelSet2D &>(*sim);
    const size_t after = sim->getManifold().springlCount();
    // Tracking error in base pixels (finest cells are 1/32 of one), comparable across depths.
    const float trackingPixels = adaptive.maxTrackingError() / 32.0f;
    std::printf("  depth 5: springls %zu -> %zu, max |phi| at particles %.2f base pixels\n", initial, after, trackingPixels);
    CHECK(after < initial * 13 / 10);  // fixed: 2,413 -> 2,906 (depth 4: 1,209 -> 1,373); broken: 8,725
    CHECK(trackingPixels < 0.6f);      // fixed: 0.37 (depth 4: 0.43); broken: 0.94
}

TEST(adaptive_split_keeps_label)
{
    const EnrightConfig2D config = Config(32);
    SplitProbe sim;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    const size_t before = sim.getManifold().springlCount();
    const int label = sim.getManifold().springlLabels[0];
    sim.splitSpringl(0);
    const Manifold2D &m = sim.getManifold();
    CHECK(m.springlCount() == before + 1);
    CHECK(m.springlLabels.size() == m.springlCount());
    CHECK(m.springlLabels[0] == label);
    CHECK(m.springlLabels.back() == label);
}
