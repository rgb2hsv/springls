#include "Demos.hpp"

namespace toybox::demos
{
Demo Toybox()
{
    return {"toybox",
            "Runs any implemented configuration of the springls library, chosen by representation, domain, grid type, objects, "
            "scenario and interface tracking (or by method key with -m); --list prints the implemented ones.",
            {"-m", "all"}};
}

Demo ForEntry(const Entry &entry)
{
    const Configuration &c = entry.config;
    const std::string scenario = c.scenario == Scenario::Enright ? "the Enright deformation test"
                                 : c.scenario == Scenario::Mesh  ? "a PLY mesh (relaxing and resampling, no flow)"
                                 : c.scenario == Scenario::Stone ? "the stone, a lumpy distorted icosphere (an object travels a geodesic loop)"
                                                                 : "a triangle-mesh surface (an object travels a geodesic loop)";
    return {ProgramName(entry),
            entry.title + " on " + scenario + " (" + Describe(c) + "); the same as toybox --domain " + Key(c.domain) + " --scenario " + Key(c.scenario) +
                " -m " + entry.key + ".",
            {"--domain", Key(c.domain), "--scenario", Key(c.scenario), "-m", entry.key}};
}

std::vector<Demo> All()
{
    std::vector<Demo> all = {Toybox()};
    for (const Entry &e : Registry::Instance().entries()) all.push_back(ForEntry(e));
    return all;
}

const Demo *Find(const std::string &program)
{
    static const std::vector<Demo> all = All();
    for (const Demo &p : all) {
        if (p.program == program) return &p;
    }
    return nullptr;
}
}  // namespace toybox::demos
