// Checkpoints (SpringLevelSet3D::saveState / loadState): springls in cereal, level sets in a VDB file.
#include "Check.hpp"

#include <io/Vdb.hpp>
#include <scenarios/EnrightTest3D.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

using namespace springls;

namespace
{
EnrightConfig3D SmallConfig(Method3D method)
{
    EnrightConfig3D config;
    config.gridSize = IsAdaptive(method) ? 17 : 33;
    config.radius = 0.2f;
    return config;
}

std::unique_ptr<SpringLevelSet3D> Start(Method3D method, const EnrightConfig3D &config)
{
    auto sim = MakeSimulation(method, ElementType::Quad, 1);
    ConfigureEnright(*sim, config);
    sim->init();
    return sim;
}

float MaxDifference(const Points3 &a, const Points3 &b)
{
    if (a.size() != b.size()) return 1e30f;
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); i++) worst = std::max(worst, (a[i] - b[i]).norm());
    return worst;
}
}  // namespace

TEST(checkpoint_round_trip_every_method)
{
    std::vector<Method3D> methods(kAllMethods3D.begin(), kAllMethods3D.end());
    methods.insert(methods.end(), kLevelSetMethods3D.begin(), kLevelSetMethods3D.end());
    for (Method3D method : methods) {
        const EnrightConfig3D config = SmallConfig(method);
        auto original = Start(method, config);
        for (int i = 0; i < 3; i++) original->step();

        std::stringstream state, levelSets;
        std::string error;
        CHECK(original->saveState(state, levelSets, error));
        CHECK(error.empty());
        // The level sets are a VDB file that VdbReader reads back.
        {
            std::istringstream copy(levelSets.str());
            VdbReader reader;
            CHECK(reader.read(copy, error));
            CHECK(reader.find("unsigned") != nullptr);
            if (IsAdaptive(method)) CHECK(reader.find("blocks_d0") != nullptr && (reader.find("phi_d0") || reader.find("phi_d1")));
            else CHECK(reader.find("phi") != nullptr);
        }

        auto restored = Start(method, config);
        const bool loaded = restored->loadState(state, levelSets, error);
        if (!loaded) std::printf("%s: %s\n", MethodKey(method), error.c_str());
        CHECK(loaded);
        CHECK(restored->getIteration() == original->getIteration());
        CHECK(restored->getSimulationTime() == original->getSimulationTime());
        CHECK(restored->getManifold().springlParticles == original->getManifold().springlParticles);
        CHECK(restored->getManifold().isoSurface.vertexes == original->getManifold().isoSurface.vertexes);
        CHECK(restored->getLevelSet().size() == original->getLevelSet().size());
        CHECK(std::equal(restored->getLevelSet().data(), restored->getLevelSet().data() + restored->getLevelSet().size(), original->getLevelSet().data()));

        // The restored run steps like the original.
        for (int i = 0; i < 2; i++) {
            original->step();
            restored->step();
        }
        CHECK(restored->getManifold().springlCount() == original->getManifold().springlCount());
        CHECK(MaxDifference(restored->getManifold().springlParticles, original->getManifold().springlParticles) < 1e-4f);
    }
}

TEST(checkpoint_rejects_level_sets_of_another_step)
{
    const Method3D method = Method3D::AdaptiveDweave;
    const EnrightConfig3D config = SmallConfig(method);
    auto sim = Start(method, config);
    sim->step();
    std::stringstream state1, levelSets1;
    std::string error;
    CHECK(sim->saveState(state1, levelSets1, error));
    sim->step();
    std::stringstream state2, levelSets2;
    CHECK(sim->saveState(state2, levelSets2, error));

    auto restored = Start(method, config);
    CHECK(!restored->loadState(state1, levelSets2, error));
    CHECK(error.find("step") != std::string::npos);

    // A damaged level set file is refused.
    auto other = Start(method, config);
    std::string bytes = levelSets1.str();
    bytes.resize(bytes.size() / 2);
    std::stringstream damaged(bytes), state1Again(state1.str());
    error.clear();
    CHECK(!other->loadState(state1Again, damaged, error));
    CHECK(!error.empty());
}
