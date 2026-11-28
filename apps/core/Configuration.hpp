#pragma once
// The axes that select what toybox runs: representation, domain, grid type, objects, scenario and
// interface tracking. A Configuration is one point of that space; the Registry lists the
// implemented ones.
#include <string>

namespace toybox
{
enum class Representation {
    Springls,
    Dweave,
    LevelSet,
    TrackedMesh   // a closed triangle mesh tracked with adaptive remeshing (no springls)
};
enum class Domain {
    D2,
    D3,
    Surface
};
enum class GridType {
    Uniform,
    Adaptive
};
enum class Objects {
    Single,
    Multiple
};
enum class Scenario {
    Enright,
    Mesh,
    Sphere,
    Stone    // the sphere scenario on the stone (assets/model/stone.ply) instead of the icosphere
};
// Eulerian: the level set evolves on its grid toward the springls each sub-step (CFL-limited).
// Semi-Lagrangian: the level set is rebuilt from the springls each sub-step.
enum class InterfaceTracking {
    Eulerian,
    SemiLagrangian
};

struct Configuration {
    Representation representation = Representation::Springls;
    Domain domain = Domain::D2;
    GridType grid = GridType::Uniform;
    Objects objects = Objects::Single;
    Scenario scenario = Scenario::Enright;
    InterfaceTracking tracking = InterfaceTracking::Eulerian;
    bool operator==(const Configuration &) const = default;
};

// Command-line keys: springls|dweave|levelset|tracked-mesh, 2d|3d|surface, uniform|adaptive, single|multiple,
// enright|mesh|sphere|stone, eulerian|semi-lagrangian.
const char *Key(Representation value);
const char *Key(Domain value);
const char *Key(GridType value);
const char *Key(Objects value);
const char *Key(Scenario value);
const char *Key(InterfaceTracking value);
bool Parse(const std::string &key, Representation &value);
bool Parse(const std::string &key, Domain &value);
bool Parse(const std::string &key, GridType &value);
bool Parse(const std::string &key, Objects &value);
bool Parse(const std::string &key, Scenario &value);
bool Parse(const std::string &key, InterfaceTracking &value);

// Every value of each axis, in declaration order.
constexpr Representation kRepresentations[] = {Representation::Springls, Representation::Dweave, Representation::LevelSet,
                                               Representation::TrackedMesh};
constexpr Domain kDomains[] = {Domain::D2, Domain::D3, Domain::Surface};
constexpr GridType kGridTypes[] = {GridType::Uniform, GridType::Adaptive};
constexpr Objects kObjects[] = {Objects::Single, Objects::Multiple};
constexpr Scenario kScenarios[] = {Scenario::Enright, Scenario::Mesh, Scenario::Sphere, Scenario::Stone};
constexpr InterfaceTracking kInterfaceTrackings[] = {InterfaceTracking::Eulerian, InterfaceTracking::SemiLagrangian};

// Sphere and stone: an object's springls travel a geodesic loop on a triangle-mesh surface (the same
// simulation, options and window; only the default mesh differs).
inline bool Surface(Scenario value) { return value == Scenario::Sphere || value == Scenario::Stone; }

// The keys separated by spaces: "dweave 2d adaptive single enright semi-lagrangian".
std::string Describe(const Configuration &config);
}  // namespace toybox
