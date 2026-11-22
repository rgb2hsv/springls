// Checkpoints of the 2D simulations (SpringLevelSet2D::saveState / loadState): one cereal stream.
#include "Check.hpp"

#include <scenarios/EnrightTest2D.hpp>

#include <cstdio>
#include <memory>
#include <sstream>

using namespace springls;

namespace
{
constexpr Method kAll2D[] = {
    Method::Springls,        Method::AdaptiveSpringls,      Method::Dweave,        Method::AdaptiveDweave, Method::SemiLagrangianDweave, Method::Mogac,
    Method::MultiSpringls,   Method::AdaptiveMultiSpringls, Method::AdaptiveMogac, Method::MultiDweave,    Method::AdaptiveMultiDweave,  Method::LevelSet,
    Method::AdaptiveLevelSet};

EnrightConfig2D SmallConfig(Method method)
{
    EnrightConfig2D config;
    config.gridSize = IsAdaptive(method) ? 16 : 48;
    config.period = 1.0;
    return config;
}

std::unique_ptr<SpringLevelSet2D> Start(Method method, const EnrightConfig2D &config)
{
    auto sim = MakeSimulation(method, 2);
    ConfigureEnright(*sim, config);
    sim->init();
    return sim;
}

// Whether two runs are in exactly the same state (bit for bit).
bool Same(const SpringLevelSet2D &a, const SpringLevelSet2D &b)
{
    const Manifold2D &m = a.getManifold(), &n = b.getManifold();
    return a.getIteration() == b.getIteration() && a.getSimulationTime() == b.getSimulationTime() &&
           a.getAdvectionSubstepCount() == b.getAdvectionSubstepCount() && m.springlParticles == n.springlParticles && m.springlVertexes == n.springlVertexes &&
           m.springlLabels == n.springlLabels && m.springlConstellations == n.springlConstellations && m.vertexLocations == n.vertexLocations &&
           m.vertexIndexes == n.vertexIndexes && m.curveLabels == n.curveLabels;
}
}  // namespace

TEST(checkpoint2d_resume_reproduces_every_method)
{
    for (Method method : kAll2D) {
        const EnrightConfig2D config = SmallConfig(method);
        auto original = Start(method, config);
        for (int i = 0; i < 3; i++) original->step();

        std::stringstream state;
        std::string error;
        CHECK(original->saveState(state, error));
        CHECK(error.empty());

        auto restored = Start(method, config);
        const bool loaded = restored->loadState(state, error);
        if (!loaded) std::printf("  %s: %s\n", MethodKey(method), error.c_str());
        CHECK(loaded);
        CHECK(Same(*original, *restored));

        // The resumed run continues exactly as the uninterrupted one.
        for (int i = 0; i < 4; i++) {
            original->step();
            restored->step();
        }
        const bool same = Same(*original, *restored);
        if (!same) std::printf("  %s: resumed run diverged\n", MethodKey(method));
        CHECK(same);
    }
}

TEST(checkpoint2d_rejects_another_method)
{
    const EnrightConfig2D config = SmallConfig(Method::Springls);
    auto springls = Start(Method::Springls, config);
    springls->step();
    std::stringstream state;
    std::string error;
    CHECK(springls->saveState(state, error));
    auto dweave = Start(Method::Dweave, config);
    CHECK(!dweave->loadState(state, error));
    CHECK(error.find("not") != std::string::npos);
}

TEST(checkpoint2d_rejects_garbage)
{
    auto sim = Start(Method::Springls, SmallConfig(Method::Springls));
    std::istringstream junk("this is not a checkpoint");
    std::string error;
    CHECK(!sim->loadState(junk, error));
    CHECK(!error.empty());
}
