// MultiSpringLevelSet2D: springls on the MOGAC label + unsigned distance band (fixed grid).
#include "Check.hpp"

#include <scenarios/EnrightTest2D.hpp>
#include <springl/MultiSpringLevelSet2D.hpp>

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

std::unique_ptr<SpringLevelSet2D> Initialized(Method method, const EnrightConfig2D &config)
{
    auto sim = MakeSimulation(method, 2);
    CHECK(sim != nullptr);
    if (!sim) return nullptr;
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    return sim;
}

EnrightMetrics2D RunToEnd(Method method, const EnrightConfig2D &config)
{
    auto sim = Initialized(method, config);
    if (!sim) return {};
    int steps = 0;
    while (sim->step() && steps < 100000) steps++;
    return MeasureEnright(*sim, config);
}
}  // namespace

TEST(multi_springls_seeds_labeled_springls)
{
    auto sim = Initialized(Method::MultiSpringls, Config(64));
    if (!sim) return;
    const Manifold2D &m = sim->getManifold();
    CHECK(m.springlCount() > 0);
    CHECK(m.springlLabels.size() == m.springlCount());
    int counts[kEnrightRegionCount + 1] = {};
    for (int l : m.springlLabels) {
        CHECK(l >= 1 && l <= kEnrightRegionCount);
        if (l >= 1 && l <= kEnrightRegionCount) counts[l]++;
    }
    for (int l = 2; l <= kEnrightRegionCount; l++) CHECK(std::abs(counts[l] - counts[1]) <= counts[1] / 4);
}

TEST(multi_springls_shared_boundary_two_layers)
{
    const EnrightConfig2D config = Config(64);
    auto sim = Initialized(Method::MultiSpringls, config);
    if (!sim) return;
    const Manifold2D &m = sim->getManifold();
    // On the boundary between quadrants 1 (above) and 4 (below): the +x half of the horizontal diameter.
    const Vec2 probe = config.centerPixels() + Vec2(0.5f * config.radiusPixels(), 0.0f);
    bool near1 = false, near4 = false;
    for (size_t n = 0; n < m.springlCount(); n++) {
        if ((m.springlParticles[n] - probe).norm() > 1.0f) continue;
        near1 |= m.springlLabels[n] == 1;
        near4 |= m.springlLabels[n] == 4;
    }
    CHECK(near1);
    CHECK(near4);
}

TEST(multi_springls_enright_headless)
{
    // Grid 64: at 32 a quarter disk (r 4.6 px) stretches thinner than a pixel, which one label per
    // pixel cannot hold. First run at 64: errors -0.053 -0.070 -0.034 -0.029 (MOGAC: -0.49 .. -0.65).
    const EnrightConfig2D config = Config(64);
    const EnrightMetrics2D springls = RunToEnd(Method::MultiSpringls, config);
    const EnrightMetrics2D mogac = RunToEnd(Method::Mogac, config);
    CHECK_NEAR(springls.time, 1.0, 1E-9);
    CHECK(springls.labelRelativeAreaErrors.size() == static_cast<size_t>(kEnrightRegionCount));
    double springlsMean = 0.0, mogacMean = 0.0;
    for (int l = 0; l < kEnrightRegionCount; l++) {
        std::printf("  label %d area error: multi springls %.4f, mogac %.4f\n", l + 1, springls.labelRelativeAreaErrors[l], mogac.labelRelativeAreaErrors[l]);
        CHECK(std::abs(springls.labelRelativeAreaErrors[l]) < 0.11);
        springlsMean += std::abs(springls.labelRelativeAreaErrors[l]) / kEnrightRegionCount;
        mogacMean += std::abs(mogac.labelRelativeAreaErrors[l]) / kEnrightRegionCount;
    }
    CHECK(springlsMean < mogacMean);
}
