// AdaptiveMultiActiveContour2D: plain MOGAC (no springls) on the labeled block-adaptive quadtree.
#include "Check.hpp"

#include <levelset/AdaptiveMultiActiveContour2D.hpp>
#include <scenarios/EnrightTest2D.hpp>

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

std::unique_ptr<SpringLevelSet2D> Initialized(const EnrightConfig2D &config, int depth)
{
    auto sim = MakeSimulation(Method::AdaptiveMogac, depth);
    CHECK(sim != nullptr);
    if (!sim) return nullptr;
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    return sim;
}
}  // namespace

TEST(adaptive_mogac_init_has_labeled_curves_and_no_springls)
{
    auto sim = Initialized(Config(32), 2);
    if (!sim) return;
    const Manifold2D &m = sim->getManifold();
    CHECK(m.springlCount() == 0);
    CHECK(m.vertexIndexes.size() >= static_cast<size_t>(kEnrightRegionCount));
    CHECK(m.curveLabels.size() == m.vertexIndexes.size());
    // The band around the region boundaries is at the finest depth.
    const AdaptiveGrid2D &grid = dynamic_cast<const AdaptiveMultiActiveContour2D &>(*sim).getGrid();
    int coarse = 0;
    for (const Vec2 &v : m.vertexLocations) coarse += (grid.blocks()[grid.findLeaf(v)].key.depth != 2) ? 1 : 0;
    CHECK(coarse == 0);
}

TEST(adaptive_mogac_enright_headless)
{
    // Grid 32 at depth 2 (128 equivalent). The uniform MOGAC at 128 loses 11-31 % per region; the
    // adaptive one must do no worse than that bound.
    const EnrightConfig2D config = Config(32);
    auto sim = Initialized(config, 2);
    if (!sim) return;
    int steps = 0;
    while (sim->step() && steps < 100000) steps++;
    const EnrightMetrics2D m = MeasureEnright(*sim, config);
    CHECK_NEAR(m.time, 1.0, 1E-9);
    CHECK(m.labelRelativeAreaErrors.size() == static_cast<size_t>(kEnrightRegionCount));
    for (size_t l = 0; l < m.labelRelativeAreaErrors.size(); l++) {
        std::printf("  adaptive mogac label %zu area error %.4f\n", l + 1, m.labelRelativeAreaErrors[l]);
        CHECK(std::abs(m.labelRelativeAreaErrors[l]) < 0.46);
    }
    // The band followed the regions: every contour vertex still lies in a finest leaf.
    const AdaptiveGrid2D &grid = dynamic_cast<const AdaptiveMultiActiveContour2D &>(*sim).getGrid();
    int coarse = 0;
    for (const Vec2 &v : sim->getManifold().vertexLocations) coarse += (grid.blocks()[grid.findLeaf(v)].key.depth != 2) ? 1 : 0;
    CHECK(coarse == 0);
}
