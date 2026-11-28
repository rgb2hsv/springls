#include "Check.hpp"

#include <core/Configuration.hpp>
#include <core/Demos.hpp>
#include <core/Registry.hpp>

#include <scenarios/EnrightTest2D.hpp>
#include <scenarios/EnrightTest3D.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace toybox;

namespace
{
size_t CountDomain(Domain domain, Scenario scenario)
{
    size_t n = 0;
    for (const Entry &e : Registry::Instance().entries()) n += (e.config.domain == domain && e.config.scenario == scenario) ? 1 : 0;
    return n;
}
}  // namespace

TEST(registry_lists_every_implemented_configuration)
{
    const Registry &registry = Registry::Instance();
    CHECK(registry.entries().size() == 24);
    CHECK(CountDomain(Domain::D2, Scenario::Enright) == 13);
    CHECK(CountDomain(Domain::D3, Scenario::Enright) == 8);
    CHECK(CountDomain(Domain::D3, Scenario::Mesh) == 1);
    CHECK(CountDomain(Domain::Surface, Scenario::Sphere) == 1);
    CHECK(CountDomain(Domain::Surface, Scenario::Stone) == 1);
    for (springls::Method m : springls::kAllMethods) {
        const Entry *e = registry.findKey(Domain::D2, Scenario::Enright, springls::MethodKey(m));
        CHECK(e != nullptr && e->title == springls::MethodTitle(m) && e->config.objects == Objects::Single);
    }
    for (springls::Method m : springls::kMultiMethods) {
        const Entry *e = registry.findKey(Domain::D2, Scenario::Enright, springls::MethodKey(m));
        CHECK(e != nullptr && e->config.objects == Objects::Multiple);
    }
    for (springls::Method m : springls::kLevelSetMethods) {
        const Entry *e = registry.findKey(Domain::D2, Scenario::Enright, springls::MethodKey(m));
        CHECK(e != nullptr && e->config.representation == Representation::LevelSet && e->config.objects == Objects::Single &&
              (e->config.grid == GridType::Adaptive) == springls::IsAdaptive(m));
    }
    for (springls::Method3D m : springls::kLevelSetMethods3D) {
        const Entry *e = registry.findKey(Domain::D3, Scenario::Enright, springls::MethodKey(m));
        CHECK(e != nullptr && e->config.representation == Representation::LevelSet && e->title == springls::MethodTitle(m) &&
              (e->config.grid == GridType::Adaptive) == springls::IsAdaptive(m));
    }
    const Entry *sl = registry.findKey(Domain::D2, Scenario::Enright, springls::MethodKey(springls::Method::SemiLagrangianDweave));
    CHECK(sl != nullptr && sl->config.tracking == InterfaceTracking::SemiLagrangian && sl->config.grid == GridType::Adaptive);
    for (springls::Method3D m : springls::kAllMethods3D) {
        const Entry *e = registry.findKey(Domain::D3, Scenario::Enright, springls::MethodKey(m));
        CHECK(e != nullptr && e->config.domain == Domain::D3 && e->title == springls::MethodTitle(m));
    }
    const Entry *mesh = registry.findKey(Domain::D3, Scenario::Mesh, "dweave");
    CHECK(mesh != nullptr && mesh->title == "Mesh DoubleWeave" && mesh->config.representation == Representation::Dweave &&
          mesh->config.grid == GridType::Adaptive);
}

TEST(registry_has_sphere_entry)
{
    const Registry &registry = Registry::Instance();
    size_t n = 0;
    const Entry *found = nullptr;
    for (const Entry &e : registry.entries()) {
        if (e.config.scenario == Scenario::Sphere) {
            n++;
            found = &e;
        }
    }
    CHECK(n == 1 && found != nullptr);
    if (!found) return;
    CHECK(ProgramName(*found) == "sphere-surf-multi-springls" && found->title == "Surface Multi-Object Springls" && found->key == "multi-springls");
    CHECK(found->config ==
          (Configuration{Representation::Springls, Domain::Surface, GridType::Uniform, Objects::Multiple, Scenario::Sphere, InterfaceTracking::Eulerian}));
    const Demo *demo = demos::Find("sphere-surf-multi-springls");
    CHECK(demo != nullptr && demo->program == "sphere-surf-multi-springls");
}

TEST(registry_has_stone_entry)
{
    // The sphere entry on the stone: the same method, its own scenario and program.
    const Registry &registry = Registry::Instance();
    const Entry *stone = registry.findKey(Domain::Surface, Scenario::Stone, "multi-springls");
    CHECK(stone != nullptr);
    if (!stone) return;
    CHECK(ProgramName(*stone) == "stone-surf-multi-springls" && stone->title == "Surface Multi-Object Springls" && stone->make);
    CHECK(stone->config ==
          (Configuration{Representation::Springls, Domain::Surface, GridType::Uniform, Objects::Multiple, Scenario::Stone, InterfaceTracking::Eulerian}));
    CHECK(Surface(Scenario::Stone) && Surface(Scenario::Sphere) && !Surface(Scenario::Mesh) && !Surface(Scenario::Enright));
    const Demo *demo = demos::Find("stone-surf-multi-springls");
    CHECK(demo != nullptr && demo->description.find("stone") != std::string::npos);
}

TEST(registry_rejects_unimplemented)
{
    const Registry &registry = Registry::Instance();
    using R = Representation;
    CHECK(registry.find({R::LevelSet, Domain::D3, GridType::Uniform, Objects::Multiple, Scenario::Enright, InterfaceTracking::Eulerian}) == nullptr);
    CHECK(registry.find({R::LevelSet, Domain::D2, GridType::Uniform, Objects::Single, Scenario::Enright, InterfaceTracking::Eulerian}) != nullptr);
    CHECK(registry.find({R::Springls, Domain::D3, GridType::Uniform, Objects::Multiple, Scenario::Enright, InterfaceTracking::Eulerian}) == nullptr);
    CHECK(registry.find({R::Springls, Domain::Surface, GridType::Uniform, Objects::Multiple, Scenario::Enright, InterfaceTracking::Eulerian}) == nullptr);
    CHECK(registry.find({R::Dweave, Domain::D2, GridType::Uniform, Objects::Single, Scenario::Enright, InterfaceTracking::Eulerian}) != nullptr);
}

TEST(registry_nearest_same_domain_and_scenario)
{
    const Configuration wanted{Representation::Springls, Domain::D3, GridType::Uniform, Objects::Multiple, Scenario::Enright, InterfaceTracking::Eulerian};
    const auto nearest = Registry::Instance().nearest(wanted);
    CHECK(!nearest.empty());
    for (const Entry *e : nearest) CHECK(e->config.domain == Domain::D3 && e->config.scenario == Scenario::Enright);
    // The closest: 3D uniform springls (differs only in objects).
    CHECK(!nearest.empty() && nearest.front()->key == "springls" && nearest.front()->config.domain == Domain::D3);
}

TEST(registry_keys_round_trip)
{
    const Registry &registry = Registry::Instance();
    const Entry *e = registry.findKey(Domain::D3, Scenario::Enright, "dweave");
    CHECK(e != nullptr && e->config.grid == GridType::Uniform && e->config.representation == Representation::Dweave && e->config.domain == Domain::D3);
    e = registry.findKey(Domain::D2, Scenario::Enright, "adaptive-multi-dweave");
    CHECK(e != nullptr && e->config.grid == GridType::Adaptive && e->config.objects == Objects::Multiple && e->title == "Adaptive Multi-Object DoubleWeave");
    CHECK(registry.findKey(Domain::D2, Scenario::Enright, "no-such-method") == nullptr);
    // find() of an entry's configuration gives that entry.
    for (const Entry &entry : registry.entries()) CHECK(registry.find(entry.config) == &entry);
}

TEST(configuration_keys_parse)
{
    for (Representation v : {Representation::Springls, Representation::Dweave, Representation::LevelSet}) {
        Representation p{};
        CHECK(Parse(Key(v), p) && p == v);
    }
    for (Domain v : {Domain::D2, Domain::D3, Domain::Surface}) {
        Domain p{};
        CHECK(Parse(Key(v), p) && p == v);
    }
    for (GridType v : {GridType::Uniform, GridType::Adaptive}) {
        GridType p{};
        CHECK(Parse(Key(v), p) && p == v);
    }
    for (Objects v : {Objects::Single, Objects::Multiple}) {
        Objects p{};
        CHECK(Parse(Key(v), p) && p == v);
    }
    for (Scenario v : {Scenario::Enright, Scenario::Mesh, Scenario::Sphere, Scenario::Stone}) {
        Scenario p{};
        CHECK(Parse(Key(v), p) && p == v);
    }
    for (InterfaceTracking v : {InterfaceTracking::Eulerian, InterfaceTracking::SemiLagrangian}) {
        InterfaceTracking p{};
        CHECK(Parse(Key(v), p) && p == v);
    }
    Domain d{};
    CHECK(!Parse("4d", d));
    CHECK(std::string(Key(InterfaceTracking::SemiLagrangian)) == "semi-lagrangian" && std::string(Key(Representation::LevelSet)) == "levelset");
    const Configuration c{Representation::Dweave, Domain::D2, GridType::Adaptive, Objects::Single, Scenario::Enright, InterfaceTracking::SemiLagrangian};
    CHECK(Describe(c) == "dweave 2d adaptive single enright semi-lagrangian");
}

TEST(registry_markdown_matrix_lists_entries)
{
    const std::string matrix = Registry::Instance().markdownMatrix();
    CHECK(matrix.rfind("| representation | domain | grid type | objects | scenario | interface tracking | method key | program | title |\n", 0) == 0);
    CHECK(matrix.find("| dweave | 3d | adaptive | single | mesh | eulerian | `dweave` | `mesh-3d-dweave` | Mesh DoubleWeave |\n") != std::string::npos);
    size_t lines = 0;
    for (char ch : matrix) lines += ch == '\n';
    CHECK(lines == 2 + Registry::Instance().entries().size());
}

TEST(registry_program_names_are_canonical_and_unique)
{
    const Registry &registry = Registry::Instance();
    std::set<std::string> names;
    for (const Entry &e : registry.entries()) {
        const std::string name = ProgramName(e);
        CHECK(name == std::string(Key(e.config.scenario)) + "-" + Key(e.config.domain) + "-" + e.key);
        names.insert(name);
    }
    CHECK(names.size() == registry.entries().size());
    CHECK(names.count("enright-2d-adaptive-multi-dweave") == 1 && names.count("enright-3d-semi-lagrangian-dweave") == 1 && names.count("mesh-3d-dweave") == 1 &&
          names.count("enright-2d-springls") == 1 && names.count("enright-3d-springls") == 1);
}

TEST(demos_one_per_entry)
{
    const std::vector<Demo> all = demos::All();
    CHECK(all.size() == Registry::Instance().entries().size() + 1);  // + toybox
    CHECK(!all.empty() && all.front().program == "toybox");
    for (const Entry &e : Registry::Instance().entries()) {
        const Demo p = demos::ForEntry(e);
        CHECK(p.program == ProgramName(e) && p.description.find(e.title) != std::string::npos);
        const Demo *found = demos::Find(p.program);
        CHECK(found != nullptr && found->program == p.program && found->defaultArgs == p.defaultArgs);
    }
    CHECK(demos::Find("enright2d") == nullptr);
}

// CMakeLists.txt builds exactly one program per registry entry (TOYBOX_PROGRAMS).
TEST(cmake_builds_one_program_per_entry)
{
    std::ifstream in(std::string(SPRINGLS_SOURCE_DIR) + "/CMakeLists.txt");
    std::stringstream text;
    text << in.rdbuf();
    const std::string cmake = text.str();
    const size_t start = cmake.find("set(TOYBOX_PROGRAMS");
    CHECK(start != std::string::npos);
    if (start == std::string::npos) return;
    const size_t end = cmake.find(')', start);
    std::istringstream list(cmake.substr(start + 19, end - start - 19));
    std::set<std::string> programs;
    std::string word;
    while (list >> word) programs.insert(word);
    std::set<std::string> expected;
    for (const Entry &e : Registry::Instance().entries()) expected.insert(ProgramName(e));
    CHECK(programs == expected);
}
