#include "Configuration.hpp"

#include <iterator>

namespace toybox
{
const char *Key(Representation value)
{
    switch (value) {
        case Representation::Springls: return "springls";
        case Representation::Dweave: return "dweave";
        case Representation::LevelSet: return "levelset";
        case Representation::TrackedMesh: return "tracked-mesh";
    }
    return "";
}

const char *Key(Domain value)
{
    switch (value) {
        case Domain::D2: return "2d";
        case Domain::D3: return "3d";
        case Domain::Surface: return "surf";
    }
    return "";
}

const char *Key(GridType value) { return value == GridType::Uniform ? "uniform" : "adaptive"; }
const char *Key(Objects value) { return value == Objects::Single ? "single" : "multiple"; }
const char *Key(Scenario value)
{
    switch (value) {
        case Scenario::Enright: return "enright";
        case Scenario::Mesh: return "mesh";
        case Scenario::Sphere: return "sphere";
        case Scenario::Stone: return "stone";
    }
    return "";
}
const char *Key(InterfaceTracking value) { return value == InterfaceTracking::Eulerian ? "eulerian" : "semi-lagrangian"; }

namespace
{
template <class T, size_t N>
bool ParseKey(const std::string &key, const T (&values)[N], T &value)
{
    for (T v : values) {
        if (key == Key(v)) {
            value = v;
            return true;
        }
    }
    return false;
}
}  // namespace

bool Parse(const std::string &key, Representation &value) { return ParseKey(key, kRepresentations, value); }
bool Parse(const std::string &key, Domain &value) { return ParseKey(key, kDomains, value); }
bool Parse(const std::string &key, GridType &value) { return ParseKey(key, kGridTypes, value); }
bool Parse(const std::string &key, Objects &value) { return ParseKey(key, kObjects, value); }
bool Parse(const std::string &key, Scenario &value) { return ParseKey(key, kScenarios, value); }
bool Parse(const std::string &key, InterfaceTracking &value) { return ParseKey(key, kInterfaceTrackings, value); }

std::string Describe(const Configuration &c)
{
    return std::string(Key(c.representation)) + " " + Key(c.domain) + " " + Key(c.grid) + " " + Key(c.objects) + " " + Key(c.scenario) + " " + Key(c.tracking);
}
}  // namespace toybox
