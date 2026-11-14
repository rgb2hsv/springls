#pragma once
#include <springls/SpringLevelSet3D.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace springls
{
// 3D Enright deformation benchmark [Enright, Fedkiw, Ferziger and Mitchell, J. Comput. Phys. 2002].
//
// Domain [0,1]^3 sampled on a gridSize^3 level set (domain x = i / (gridSize - 1)). Velocity, in
// domain units per unit time:
//   u(x,y,z,t) =  2 sin^2(pi x) sin(2 pi y) sin(2 pi z) cos(pi t / T)
//   v(x,y,z,t) = -sin(2 pi x) sin^2(pi y) sin(2 pi z) cos(pi t / T)
//   w(x,y,z,t) = -sin(2 pi x) sin(2 pi y) sin^2(pi z) cos(pi t / T)
// The flow is divergence free and reverses at t = T/2, so at t = T the exact solution is the
// initial sphere again: any volume change or shape difference is numerical error.
struct EnrightConfig3D {
    int gridSize = 128;
    double period = 3.0;
    Vec3 center = Vec3::Constant(0.35f);
    float radius = 0.15f;
    // Cap on one advection sub-step, as a fraction of the period.
    double maxTimeStepFraction = 1.0 / 800.0;

    float voxelScale() const { return static_cast<float>(gridSize - 1); }  // voxels per domain unit
    Vec3 centerVoxels() const { return center * voxelScale(); }
    float radiusVoxels() const { return radius * voxelScale(); }
};

// Signed distance to the initial sphere (negative inside), in voxels.
VolumeF MakeEnrightInitialLevelSet(const EnrightConfig3D &config);
// Spatial part of the velocity at a voxel position, in voxels per unit time (the time factor is
// the schedule weight). Exact: ConfigureEnright() advects springls with it directly.
Vec3 EnrightVelocity(const EnrightConfig3D &config, const Vec3 &position);
// The same velocity sampled on the level-set grid (for flows that only exist as fields).
VolumeV3 MakeEnrightVelocityField(const EnrightConfig3D &config);
// Configures sim for the test, with the analytic velocity (no interpolation error). Call sim.init() afterwards; step() returns false once t reaches T.
void ConfigureEnright(SpringLevelSet3D &sim, const EnrightConfig3D &config);

struct EnrightMetrics3D {
    double time = 0.0;
    double volume = 0.0;               // enclosed by the iso-surface, domain units^3
    double relativeVolumeError = 0.0;  // (volume - exact) / exact
    // Distance of iso-surface vertexes from the initial sphere, in domain units (meaningful at
    // t = 0 and t = T, when the exact interface is that sphere).
    double meanShapeError = 0.0;
    double maxShapeError = 0.0;
    size_t springlCount = 0;
    std::array<size_t, 2> constellationSpringlCounts = {0, 0};  // primal, dual
};
EnrightMetrics3D MeasureEnright(const SpringLevelSet3D &sim, const EnrightConfig3D &config);

// The 3D springl methods.
enum class Method3D {
    Springls,
    Doubleweave,
    AdaptiveSpringls,
    AdaptiveDoubleweave,
    SemiLagrangianDoubleweave
};
constexpr std::array<Method3D, 5> kAllMethods3D = {Method3D::Springls, Method3D::AdaptiveSpringls, Method3D::Doubleweave, Method3D::AdaptiveDoubleweave,
                                                   Method3D::SemiLagrangianDoubleweave};
bool IsAdaptive(Method3D method);
// Default level-set grid size: 128 for the uniform methods, a 32^3 base grid for the adaptive ones
// (refined up to 32 * 2^maxDepth, 256^3 equivalent at the default depth 3).
int DefaultGridSize(Method3D method);
// Command-line key ("springls", "doubleweave", "adaptive-springls", "adaptive-doubleweave",
// "semi-lagrangian-doubleweave").
const char *MethodKey(Method3D method);
const char *MethodTitle(Method3D method);
bool ParseMethod(const std::string &key, Method3D &method);
const char *ElementTypeKey(ElementType type);  // "quad", "triangle"
bool ParseElementType(const std::string &key, ElementType &type);
// New simulation for `method`; maxDepth applies to the adaptive methods.
std::unique_ptr<SpringLevelSet3D> MakeSimulation(Method3D method, ElementType elementType, int maxDepth = 3);
// Sets a tuning parameter by name (see TuningKeys3D()); returns false for an unknown key.
bool SetTuning(SpringLevelSet3D &sim, const std::string &key, double value);
std::vector<std::string> TuningKeys3D();
}  // namespace springls
