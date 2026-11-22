// Multi-object Enright test: quadrant labels, partition distance, per-label metrics.
#include "Check.hpp"

#include <scenarios/EnrightTest2D.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

using namespace springls;

namespace
{
struct LabeledProbe : SpringLevelSet2D {
    LabeledProbe() { mLabeled = true; }
    using SpringLevelSet2D::mManifold;
};

EnrightConfig2D SmallConfig()
{
    EnrightConfig2D config;
    config.gridSize = 32;
    config.period = 1.0;
    return config;
}

// Runs `method` through the whole period and returns the final metrics.
EnrightMetrics2D RunToEnd(Method method, const EnrightConfig2D &config)
{
    auto sim = MakeSimulation(method, 2);
    CHECK(sim != nullptr);
    if (!sim) return {};
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    int steps = 0;
    while (sim->step() && steps < 100000) steps++;
    return MeasureEnright(*sim, config);
}
}  // namespace

TEST(enright_initial_labels_quadrants)
{
    EnrightConfig2D config;
    config.gridSize = 64;
    const ImageI labels = MakeEnrightInitialLabels(config);
    const Vec2 c = config.centerPixels();
    const int cx = static_cast<int>(std::round(c.x())), cy = static_cast<int>(std::round(c.y()));
    CHECK(labels(cx + 5, cy + 5) == 1);
    CHECK(labels(cx - 5, cy + 5) == 2);
    CHECK(labels(cx - 5, cy - 5) == 3);
    CHECK(labels(cx + 5, cy - 5) == 4);
    CHECK(labels(0, 0) == 0);
    int counts[5] = {0, 0, 0, 0, 0};
    for (size_t n = 0; n < labels.size(); n++) counts[labels[n]]++;
    for (int l = 2; l <= 4; l++) CHECK(std::abs(counts[l] - counts[1]) <= 2 * config.gridSize / 8);
}

TEST(enright_partition_distance_zero_on_interfaces)
{
    EnrightConfig2D config;
    config.gridSize = 64;
    const Vec2 c = config.centerPixels();
    const float r = config.radiusPixels();
    CHECK_NEAR(EnrightPartitionDistance(config, c), 0.0f, 1E-4f);
    CHECK_NEAR(EnrightPartitionDistance(config, c + Vec2(r, 0)), 0.0f, 1E-4f);
    CHECK_NEAR(EnrightPartitionDistance(config, c + Vec2(0, 0.5f * r)), 0.0f, 1E-4f);
    CHECK_NEAR(EnrightPartitionDistance(config, c + Vec2(0.3f * r, 0.3f * r)), 0.3f * r, 1E-4f);
    // Outside the disk the diameters do not extend: distance is to the circle.
    CHECK_NEAR(EnrightPartitionDistance(config, c + Vec2(0, 1.5f * r)), 0.5f * r, 1E-4f);
    const ImageF d = MakeEnrightInitialLabelDistance(config);
    CHECK_NEAR(d(3, 3), EnrightPartitionDistance(config, Vec2(3, 3)), 1E-5f);
}

TEST(enright_label_area_open_curve)
{
    const EnrightConfig2D config = SmallConfig();
    LabeledProbe probe;
    probe.mManifold.vertexLocations = {Vec2(0, 0), Vec2(10, 0), Vec2(10, 10)};
    probe.mManifold.vertexIndexes = {{0, 1, 2}};
    probe.mManifold.curveLabels = {1};
    const EnrightMetrics2D m = MeasureEnright(probe, config);
    const double scale = config.pixelScale();
    CHECK(m.labelAreas.size() == static_cast<size_t>(kEnrightRegionCount));
    CHECK_NEAR(m.labelAreas[0], 50.0 / (scale * scale), 1E-9);
    CHECK_NEAR(m.labelAreas[1], 0.0, 1E-12);
    CHECK_NEAR(m.area, 50.0 / (scale * scale), 1E-9);
}

TEST(enright_mogac_runs_headless)
{
    // Grid 128: at 32 the quarter disks are under 5 pixels in radius and first-order Eulerian
    // advection erases them, as it does a single level set (MOGAC with one label matches
    // NarrowBand2D advection exactly). First run: errors -0.187 -0.115 -0.305 -0.244.
    EnrightConfig2D config = SmallConfig();
    config.gridSize = 128;
    const EnrightMetrics2D m = RunToEnd(Method::Mogac, config);
    CHECK_NEAR(m.time, 1.0, 1E-9);
    CHECK(m.labelRelativeAreaErrors.size() == static_cast<size_t>(kEnrightRegionCount));
    for (size_t l = 0; l < m.labelRelativeAreaErrors.size(); l++) CHECK(std::abs(m.labelRelativeAreaErrors[l]) < 0.46);
}

TEST(enright_default_grid_sizes)
{
    // Uniform-grid methods default to 256 x 256; adaptive ones to a 32 x 32 base grid (512 x 512
    // equivalent at depth 4).
    for (Method m : {Method::Springls, Method::Dweave, Method::Mogac, Method::MultiSpringls}) CHECK(DefaultGridSize(m) == 256);
    for (Method m : {Method::AdaptiveSpringls, Method::AdaptiveDweave, Method::SemiLagrangianDweave, Method::AdaptiveMultiSpringls}) {
        CHECK(DefaultGridSize(m) == 32);
    }
}

TEST(enright_multi_springls_seed_evenly)
{
    // Springls are seeded one per initial contour segment: the multi-object contour's spacing is
    // evened out as the single-object one is, so no seeded springl is a sliver of a grid crossing.
    // (DoubleWeave's dual springls are chords across contour vertexes, which cut region junction
    // corners short: only the primal ones are measured.)
    for (Method method : {Method::Springls, Method::MultiSpringls, Method::MultiDweave, Method::AdaptiveMultiSpringls, Method::AdaptiveMultiDweave}) {
        EnrightConfig2D config;
        config.gridSize = IsAdaptive(method) ? 32 : 128;
        auto sim = MakeSimulation(method, 2);
        ConfigureEnright(*sim, config);
        CHECK(sim->init());
        const Manifold2D &m = sim->getManifold();
        size_t N = 0;
        double sum = 0.0, shortest = 1e9;
        for (size_t i = 0; i < m.springlParticles.size(); i++) {
            if (!m.springlConstellations.empty() && m.springlConstellations[i] != static_cast<uint8_t>(Constellation::Primal)) continue;
            N++;
            const double length = (m.springlVertexes[2 * i + 1] - m.springlVertexes[2 * i]).norm();
            sum += length;
            shortest = std::min(shortest, length);
        }
        std::printf("  %s: %zu springls, shortest %.3f, mean %.3f\n", MethodKey(method), N, shortest, sum / N);
        CHECK(N > 0 && shortest > 0.6 * sum / N);
    }
}

TEST(enright_one_region_matches_single_object)
{
    // The whole disk as one labeled region: the multi-object framework must track it as the
    // single-object one does (same area, shape and springl count, up to small differences).
    EnrightConfig2D config;
    config.gridSize = 64;
    config.period = 1.0;
    config.regions = 1;
    const ImageI labels = MakeEnrightInitialLabels(config);
    int inside = 0;
    for (size_t n = 0; n < labels.size(); n++) inside += (labels[n] == 1) ? 1 : (labels[n] == 0 ? 0 : 1000);
    CHECK(inside > 0 && inside < 1000);
    const std::pair<Method, Method> pairs[] = {{Method::Springls, Method::MultiSpringls},
                                               {Method::Dweave, Method::MultiDweave},
                                               {Method::AdaptiveSpringls, Method::AdaptiveMultiSpringls},
                                               {Method::AdaptiveDweave, Method::AdaptiveMultiDweave}};
    for (const auto &[single, multi] : pairs) {
        EnrightConfig2D c = config;
        if (IsAdaptive(single)) c.gridSize = 16;
        const EnrightMetrics2D a = RunToEnd(single, c), b = RunToEnd(multi, c);
        std::printf("  %s vs %s: area %.4f / %.4f, mean shape %.5f / %.5f, springls %zu / %zu\n", MethodKey(single), MethodKey(multi), a.relativeAreaError,
                    b.relativeAreaError, a.meanShapeError, b.meanShapeError, a.springlCount, b.springlCount);
        CHECK(b.labelRelativeAreaErrors.size() == 1);
        CHECK(std::abs(a.relativeAreaError - b.relativeAreaError) < 0.005);
        CHECK(std::abs(a.meanShapeError - b.meanShapeError) < 0.001);
        CHECK(std::abs(static_cast<double>(a.springlCount) - static_cast<double>(b.springlCount)) <= 0.1 * a.springlCount);
    }
}
