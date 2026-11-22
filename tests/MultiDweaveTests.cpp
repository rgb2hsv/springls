// Multi-object DoubleWeave springls (uniform and adaptive): every springl carries a region label
// and a constellation; springls interact only within both.
#include "Check.hpp"

#include <dweave/AdaptiveMultiDweaveLevelSet2D.hpp>
#include <dweave/MultiDweaveLevelSet2D.hpp>
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

struct UniformProbe : MultiDweaveLevelSet2D {
    using MultiDweaveLevelSet2D::mNearestNeighbor;
    using MultiDweaveLevelSet2D::updateNearestNeighbors;
};

struct AdaptiveProbe : AdaptiveMultiDweaveLevelSet2D {
    using AdaptiveMultiDweaveLevelSet2D::splitSpringl;
};

// Every label 1..4 has springls of both constellations.
void CheckSeeded(const Manifold2D &m)
{
    CHECK(m.springlLabels.size() == m.springlCount());
    CHECK(m.springlConstellations.size() == m.springlCount());
    int counts[kEnrightRegionCount + 1][kConstellationCount] = {};
    for (size_t n = 0; n < m.springlCount(); n++) {
        const int l = m.springlLabels[n];
        CHECK(l >= 1 && l <= kEnrightRegionCount);
        if (l >= 1 && l <= kEnrightRegionCount) counts[l][m.springlConstellations[n]]++;
    }
    for (int l = 1; l <= kEnrightRegionCount; l++) {
        CHECK(counts[l][0] > 0);
        CHECK(counts[l][1] > 0);
    }
}

EnrightMetrics2D RunToEnd(SpringLevelSet2D &sim, const EnrightConfig2D &config)
{
    int steps = 0;
    while (sim.step() && steps < 100000) steps++;
    return MeasureEnright(sim, config);
}
}  // namespace

TEST(multi_dweave_seeds_both_constellations_per_label)
{
    const EnrightConfig2D config = Config(64);
    MultiDweaveLevelSet2D sim;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    CheckSeeded(sim.getManifold());
}

TEST(multi_dweave_neighbors_share_label_and_constellation)
{
    const EnrightConfig2D config = Config(64);
    UniformProbe sim;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    for (int k = 0; k < 5; k++) sim.step();
    sim.updateNearestNeighbors();
    const Manifold2D &m = sim.getManifold();
    int linked = 0, mixed = 0;
    for (size_t e = 0; e < sim.mNearestNeighbor.size(); e++) {
        const int nbr = sim.mNearestNeighbor[e];
        if (nbr < 0) continue;
        linked++;
        const size_t a = e / 2, b = static_cast<size_t>(nbr) / 2;
        if (m.springlLabels[a] != m.springlLabels[b] || m.springlConstellations[a] != m.springlConstellations[b]) mixed++;
    }
    CHECK(linked > 0);
    CHECK(mixed == 0);
}

TEST(multi_dweave_enright_headless)
{
    const EnrightConfig2D config = Config(64);
    MultiDweaveLevelSet2D sim;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    const EnrightMetrics2D m = RunToEnd(sim, config);
    CHECK_NEAR(m.time, 1.0, 1E-9);
    CHECK(m.constellationSpringlCounts[0] > 0 && m.constellationSpringlCounts[1] > 0);
    CHECK(m.labelRelativeAreaErrors.size() == static_cast<size_t>(kEnrightRegionCount));
    for (size_t l = 0; l < m.labelRelativeAreaErrors.size(); l++) {
        std::printf("  multi DoubleWeave label %zu area error %.4f\n", l + 1, m.labelRelativeAreaErrors[l]);
        CHECK(std::abs(m.labelRelativeAreaErrors[l]) < 0.05);  // first run: -0.025 -0.033 -0.018 -0.019
    }
}

TEST(adaptive_multi_dweave_seeds_and_runs)
{
    const EnrightConfig2D config = Config(32);
    AdaptiveMultiDweaveLevelSet2D sim;
    sim.adaptiveTuning.maxDepth = 2;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    CheckSeeded(sim.getManifold());
    const EnrightMetrics2D m = RunToEnd(sim, config);
    CHECK_NEAR(m.time, 1.0, 1E-9);
    for (size_t l = 0; l < m.labelRelativeAreaErrors.size(); l++) {
        std::printf("  adaptive multi DoubleWeave label %zu area error %.4f\n", l + 1, m.labelRelativeAreaErrors[l]);
        CHECK(std::abs(m.labelRelativeAreaErrors[l]) < 0.06);  // first run: -0.035 -0.034 -0.007 -0.007
    }
}

TEST(adaptive_multi_dweave_split_keeps_label_and_constellation)
{
    const EnrightConfig2D config = Config(32);
    AdaptiveProbe sim;
    sim.adaptiveTuning.maxDepth = 2;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    const Manifold2D &m = sim.getManifold();
    // A dual springl, so the constellation is not the default.
    size_t index = 0;
    while (index < m.springlCount() && m.springlConstellations[index] != static_cast<uint8_t>(Constellation::Dual)) index++;
    CHECK(index < m.springlCount());
    if (index >= m.springlCount()) return;
    const int label = m.springlLabels[index];
    sim.splitSpringl(index);
    CHECK(m.springlLabels.size() == m.springlCount() && m.springlConstellations.size() == m.springlCount());
    CHECK(m.springlLabels.back() == label);
    CHECK(m.springlConstellations.back() == static_cast<uint8_t>(Constellation::Dual));
}

TEST(multi_dweave_methods_registered)
{
    Method m;
    CHECK(ParseMethod("multi-dweave", m) && m == Method::MultiDweave);
    CHECK(ParseMethod("adaptive-multi-dweave", m) && m == Method::AdaptiveMultiDweave);
    CHECK(DefaultGridSize(Method::MultiDweave) == kDefaultUniformGridSize);
    CHECK(DefaultGridSize(Method::AdaptiveMultiDweave) == kDefaultAdaptiveGridSize);
    CHECK(MakeSimulation(Method::AdaptiveMultiDweave, 3) != nullptr);
}
