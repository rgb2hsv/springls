#pragma once
#include <springl/SpringLevelSet2D.hpp>

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
    int gridSize = 256;
    double period = 8.0;
    Vec2 center = Vec2(0.5f, 0.75f);
    float radius = 0.15f;
    // Cap on one advection sub-step, as a fraction of the period; 0: none (the CFL limit over each
    // sub-step alone sets it).
    double maxTimeStepFraction = 0.0;
    // Multi-object methods: regions of the disk, kEnrightRegionCount quadrants or 1 (the whole disk
    // as one labeled region, to compare the multi-object framework with the single-object one).
    int regions = 4;

    float pixelScale() const { return static_cast<float>(gridSize - 1); }  // pixels per domain unit
    Vec2 centerPixels() const { return center * pixelScale(); }
    float radiusPixels() const { return radius * pixelScale(); }
};

// Signed distance to the initial circle (negative inside), in pixels.
ImageF MakeEnrightInitialLevelSet(const EnrightConfig2D &config);

// Multi-object variant: the disk split into four quadrant regions by its horizontal and vertical
// diameters (labels 1..4 counter-clockwise from the +x,+y quadrant; background 0). The shared
// boundaries are stretched by the vortex and, like the circle, must return at t = T. With
// config.regions == 1 the whole disk is region 1.
constexpr int kEnrightRegionCount = 4;
ImageI MakeEnrightInitialLabels(const EnrightConfig2D &config);
// Distance (pixels) from a point to the initial region boundaries: the circle and both diameters.
float EnrightPartitionDistance(const EnrightConfig2D &config, const Vec2 &position);
// EnrightPartitionDistance at every pixel (the unsigned distance of the initial labels).
ImageF MakeEnrightInitialLabelDistance(const EnrightConfig2D &config);
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
    // Multi-object methods only (empty otherwise), index = label - 1: area of each region's
    // curves (domain units^2) and its error against the exact region (a quarter disk, or the disk
    // with one region). For them `area` is the sum over regions and the shape errors are
    // distances to the initial region boundaries.
    std::vector<double> labelAreas;
    std::vector<double> labelRelativeAreaErrors;
};
EnrightMetrics2D MeasureEnright(const SpringLevelSet2D &sim, const EnrightConfig2D &config);

// The springl methods compared by the Enright test.
enum class Method {
    Springls,
    AdaptiveSpringls,
    Dweave,
    AdaptiveDweave,
    SemiLagrangianDweave,
    Mogac,
    MultiSpringls,
    AdaptiveMultiSpringls,
    AdaptiveMogac,
    MultiDweave,
    AdaptiveMultiDweave,
    LevelSet,
    AdaptiveLevelSet
};
// Default level-set grid of a method when none is given: 256 x 256 for the uniform-grid methods;
// 32 x 32 base grid for the adaptive ones (1024 x 1024 equivalent at the toybox default depth 5).
constexpr int kDefaultUniformGridSize = 256;
constexpr int kDefaultAdaptiveGridSize = 32;
bool IsAdaptive(Method method);
int DefaultGridSize(Method method);
// The multi-object methods (quadrant-split disk), run by enright2d with "-m multi" or by key.
constexpr std::array<Method, 6> kMultiMethods = {Method::Mogac,       Method::AdaptiveMogac,      Method::MultiSpringls, Method::AdaptiveMultiSpringls,
                                                 Method::MultiDweave, Method::AdaptiveMultiDweave};
// The basic level set method (no springls) on the uniform and the adaptive grid.
constexpr std::array<Method, 2> kLevelSetMethods = {Method::LevelSet, Method::AdaptiveLevelSet};
// The CFL-limited methods, which enright2d runs and accepts by key. The semi-Lagrangian
// DoubleWeave is run by sl-enright2d instead (MakeSimulation still builds it).
constexpr std::array<Method, 4> kAllMethods = {Method::Springls, Method::AdaptiveSpringls, Method::Dweave, Method::AdaptiveDweave};
// Command-line key ("springls", "adaptive-springls", "DoubleWeave", "adaptive-dweave",
// "semi-lagrangian-dweave", "levelset", "adaptive-levelset", ...).
const char *MethodKey(Method method);
const char *MethodTitle(Method method);
// Parses the key of one of kAllMethods, kMultiMethods or kLevelSetMethods.
bool ParseMethod(const std::string &key, Method &method);
// New simulation for `method`; maxDepth applies to the adaptive methods.
std::unique_ptr<SpringLevelSet2D> MakeSimulation(Method method, int maxDepth);
// Sets a tuning parameter by name (see TuningKeys()); returns false for an unknown key.
bool SetTuning(SpringLevelSet2D &sim, const std::string &key, double value);
std::vector<std::string> TuningKeys();
}  // namespace springls
