#pragma once
#include <springls/SpringLevelSet2D.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace springls
{
// 2D Enright deformation ("single vortex") benchmark [Enright, Fedkiw, Ferziger and Mitchell,
// "A Hybrid Particle Level Set Method for Improved Interface Capturing", J. Comput. Phys. 2002].
//
// Domain [0,1]^2 sampled on a gridSize x gridSize level set (domain x = i / (gridSize - 1)).
// Velocity, in domain units per unit time:
//   u(x,y,t) =  cos(pi t / T) sin^2(pi x) sin(2 pi y)
//   v(x,y,t) = -cos(pi t / T) sin^2(pi y) sin(2 pi x)
// The flow reverses at t = T/2, so at t = T the exact solution is the initial circle again: any
// area change or shape difference is numerical error.
struct EnrightConfig2D {
    int gridSize = 128;
    double period = 8.0;
    Vec2 center = Vec2(0.5f, 0.75f);
    float radius = 0.15f;
    // Cap on one advection sub-step, as a fraction of the period.
    double maxTimeStepFraction = 1.0 / 800.0;

    float pixelScale() const { return static_cast<float>(gridSize - 1); }  // pixels per domain unit
    Vec2 centerPixels() const { return center * pixelScale(); }
    float radiusPixels() const { return radius * pixelScale(); }
};

// Signed distance to the initial circle (negative inside), in pixels.
ImageF MakeEnrightInitialLevelSet(const EnrightConfig2D &config);
// Spatial part of the velocity at a pixel position, in pixels per unit time (the time factor is
// the schedule weight). Exact: ConfigureEnright() advects springls with it directly.
Vec2 EnrightVelocity(const EnrightConfig2D &config, const Vec2 &position);
// The same velocity sampled on the level-set grid (for flows that only exist as fields).
ImageV2 MakeEnrightVelocityField(const EnrightConfig2D &config);
// Configures sim for the test, with the analytic velocity (no interpolation error). Call sim.init() afterwards; step() returns false once t reaches T.
void ConfigureEnright(SpringLevelSet2D &sim, const EnrightConfig2D &config);

struct EnrightMetrics2D {
    double time = 0.0;
    double area = 0.0;               // enclosed by the iso-contour, domain units^2
    double relativeAreaError = 0.0;  // (area - exact) / exact
    // Distance of iso-contour vertexes from the initial circle, in domain units (meaningful at
    // t = 0 and t = T, when the exact interface is that circle).
    double meanShapeError = 0.0;
    double maxShapeError = 0.0;
    size_t springlCount = 0;
    std::array<size_t, 2> constellationSpringlCounts = {0, 0};  // primal, dual
};
EnrightMetrics2D MeasureEnright(const SpringLevelSet2D &sim, const EnrightConfig2D &config);

// The springl methods compared by the Enright test.
enum class Method {
    Springls,
    AdaptiveSpringls,
    Doubleweave,
    AdaptiveDoubleweave,
    SemiLagrangianDoubleweave
};
// The CFL-limited methods, which enright2d runs and accepts by key. The semi-Lagrangian
// doubleweave is run by sl-enright2d instead (MakeSimulation still builds it).
constexpr std::array<Method, 4> kAllMethods = {Method::Springls, Method::AdaptiveSpringls, Method::Doubleweave, Method::AdaptiveDoubleweave};
// Command-line key ("springls", "adaptive-springls", "doubleweave", "adaptive-doubleweave",
// "semi-lagrangian-doubleweave").
const char *MethodKey(Method method);
const char *MethodTitle(Method method);
// Parses the key of one of kAllMethods.
bool ParseMethod(const std::string &key, Method &method);
// New simulation for `method`; maxDepth applies to the adaptive methods.
std::unique_ptr<SpringLevelSet2D> MakeSimulation(Method method, int maxDepth);
// Sets a tuning parameter by name (see TuningKeys()); returns false for an unknown key.
bool SetTuning(SpringLevelSet2D &sim, const std::string &key, double value);
std::vector<std::string> TuningKeys();
}  // namespace springls
