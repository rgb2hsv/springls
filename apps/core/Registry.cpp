#include "Registry.hpp"

#include "Enright2DSimulation.hpp"
#include "Enright3DSimulation.hpp"
#include "MeshSimulation.hpp"
#include "SphereSimulation.hpp"

#include <algorithm>

namespace toybox
{
using springls::Method;
using springls::Method3D;

namespace
{
Configuration Config2D(Method m)
{
    Configuration c;
    c.domain = Domain::D2;
    switch (m) {
        case Method::Springls: c.representation = Representation::Springls; break;
        case Method::AdaptiveSpringls: c = {Representation::Springls, Domain::D2, GridType::Adaptive}; break;
        case Method::Dweave: c.representation = Representation::Dweave; break;
        case Method::AdaptiveDweave: c = {Representation::Dweave, Domain::D2, GridType::Adaptive}; break;
        case Method::SemiLagrangianDweave:
            c = {Representation::Dweave, Domain::D2, GridType::Adaptive, Objects::Single, Scenario::Enright, InterfaceTracking::SemiLagrangian};
            break;
        case Method::Mogac: c = {Representation::LevelSet, Domain::D2, GridType::Uniform, Objects::Multiple}; break;
        case Method::AdaptiveMogac: c = {Representation::LevelSet, Domain::D2, GridType::Adaptive, Objects::Multiple}; break;
        case Method::MultiSpringls: c = {Representation::Springls, Domain::D2, GridType::Uniform, Objects::Multiple}; break;
        case Method::AdaptiveMultiSpringls: c = {Representation::Springls, Domain::D2, GridType::Adaptive, Objects::Multiple}; break;
        case Method::MultiDweave: c = {Representation::Dweave, Domain::D2, GridType::Uniform, Objects::Multiple}; break;
        case Method::AdaptiveMultiDweave: c = {Representation::Dweave, Domain::D2, GridType::Adaptive, Objects::Multiple}; break;
        case Method::LevelSet: c.representation = Representation::LevelSet; break;
        case Method::AdaptiveLevelSet: c = {Representation::LevelSet, Domain::D2, GridType::Adaptive}; break;
    }
    return c;
}

Configuration Config3D(Method3D m)
{
    Configuration c;
    c.domain = Domain::D3;
    switch (m) {
        case Method3D::Springls:
        case Method3D::AdaptiveSpringls: c.representation = Representation::Springls; break;
        case Method3D::LevelSet:
        case Method3D::AdaptiveLevelSet: c.representation = Representation::LevelSet; break;
        case Method3D::Dweave:
        case Method3D::AdaptiveDweave:
        case Method3D::SemiLagrangianDweave: c.representation = Representation::Dweave; break;
    }
    c.grid = springls::IsAdaptive(m) ? GridType::Adaptive : GridType::Uniform;
    if (m == Method3D::SemiLagrangianDweave) c.tracking = InterfaceTracking::SemiLagrangian;
    return c;
}

int Matches(const Configuration &a, const Configuration &b)
{
    return (a.representation == b.representation) + (a.domain == b.domain) + (a.grid == b.grid) + (a.objects == b.objects) + (a.scenario == b.scenario) +
           (a.tracking == b.tracking);
}
}  // namespace

Registry::Registry()
{
    // 2D Enright: the CFL-limited single-object methods, the basic level set methods, the semi-Lagrangian
    // DoubleWeave, the multi-object methods.
    std::vector<Method> methods2D(springls::kAllMethods.begin(), springls::kAllMethods.end());
    methods2D.insert(methods2D.end(), springls::kLevelSetMethods.begin(), springls::kLevelSetMethods.end());
    methods2D.push_back(Method::SemiLagrangianDweave);
    methods2D.insert(methods2D.end(), springls::kMultiMethods.begin(), springls::kMultiMethods.end());
    for (Method m : methods2D) {
        Entry e;
        e.config = Config2D(m);
        e.key = springls::MethodKey(m);
        e.title = springls::MethodTitle(m);
        e.method2D = m;
        mEntries.push_back(e);
    }
    std::vector<Method3D> methods3D(springls::kAllMethods3D.begin(), springls::kAllMethods3D.end());
    methods3D.insert(methods3D.end(), springls::kLevelSetMethods3D.begin(), springls::kLevelSetMethods3D.end());
    for (Method3D m : methods3D) {
        Entry e;
        e.config = Config3D(m);
        e.key = springls::MethodKey(m);
        e.title = springls::MethodTitle(m);
        e.method3D = m;
        mEntries.push_back(e);
    }
    Entry mesh;
    mesh.config = {Representation::Dweave, Domain::D3, GridType::Adaptive, Objects::Single, Scenario::Mesh, InterfaceTracking::Eulerian};
    mesh.key = "dweave";  // the mesh scenario's only DoubleWeave (adaptive)
    mesh.title = "Mesh DoubleWeave";
    mesh.method3D = Method3D::AdaptiveDweave;
    mEntries.push_back(mesh);
    Entry sphere;
    sphere.config = {Representation::Springls, Domain::Surface, GridType::Uniform, Objects::Multiple, Scenario::Sphere, InterfaceTracking::Eulerian};
    sphere.key = "multi-springls";
    sphere.title = "Surface Multi-Object Springls";
    mEntries.push_back(sphere);
    Entry stone = sphere;
    stone.config.scenario = Scenario::Stone;
    mEntries.push_back(stone);

    // The factories refer to their entry, so they are set once the list is complete.
    for (Entry &e : mEntries) {
        const Entry *entry = &e;
        if (Surface(e.config.scenario)) {
            e.make = [entry](const RunOptions &options) -> std::unique_ptr<Simulation> { return std::make_unique<SphereSimulation>(*entry, options.sphere); };
        } else if (e.config.scenario == Scenario::Mesh) {
            e.make = [entry](const RunOptions &options) -> std::unique_ptr<Simulation> {
                MeshOptions mesh = options.mesh;
                mesh.elementType = options.elementType;
                mesh.tuning = options.tuning;
                return std::make_unique<MeshSimulation>(*entry, mesh);
            };
        } else if (e.config.domain == Domain::D3) {
            e.make = [entry](const RunOptions &options) -> std::unique_ptr<Simulation> { return std::make_unique<Enright3DSimulation>(*entry, options); };
        } else {
            e.make = [entry](const RunOptions &options) -> std::unique_ptr<Simulation> { return std::make_unique<Enright2DSimulation>(*entry, options); };
        }
    }
}

std::string ProgramName(const Entry &entry) { return std::string(Key(entry.config.scenario)) + "-" + Key(entry.config.domain) + "-" + entry.key; }

const Registry &Registry::Instance()
{
    static const Registry registry;
    return registry;
}

const Entry *Registry::find(const Configuration &config) const
{
    for (const Entry &e : mEntries) {
        if (e.config == config) return &e;
    }
    return nullptr;
}

const Entry *Registry::findKey(Domain domain, Scenario scenario, const std::string &key) const
{
    for (const Entry &e : mEntries) {
        if (e.config.domain == domain && e.config.scenario == scenario && e.key == key) return &e;
    }
    return nullptr;
}

std::vector<const Entry *> Registry::nearest(const Configuration &config) const
{
    std::vector<const Entry *> candidates;
    for (const Entry &e : mEntries) {
        if (e.config.domain == config.domain && e.config.scenario == config.scenario) candidates.push_back(&e);
    }
    if (candidates.empty()) {
        for (const Entry &e : mEntries) candidates.push_back(&e);
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](const Entry *a, const Entry *b) { return Matches(a->config, config) > Matches(b->config, config); });
    if (candidates.size() > 3) candidates.resize(3);
    return candidates;
}

std::string Registry::markdownMatrix() const
{
    std::string s = "| representation | domain | grid type | objects | scenario | interface tracking | method key | program | title |\n";
    s += "|---|---|---|---|---|---|---|---|---|\n";
    for (const Entry &e : mEntries) {
        const Configuration &c = e.config;
        s += std::string("| ") + Key(c.representation) + " | " + Key(c.domain) + " | " + Key(c.grid) + " | " + Key(c.objects) + " | " + Key(c.scenario) +
             " | " + Key(c.tracking) + " | `" + e.key + "` | `" + ProgramName(e) + "` | " + e.title + " |\n";
    }
    return s;
}
}  // namespace toybox
