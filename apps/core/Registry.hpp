#pragma once
// The implemented configurations: the single source for dispatch, "not implemented" errors,
// `toybox --list` and the README matrix.
#include "Configuration.hpp"

#include <scenarios/EnrightTest2D.hpp>
#include <scenarios/EnrightTest3D.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace toybox
{
class Simulation;
struct RunOptions;

struct Entry {
    Configuration config;
    std::string key;    // -m key within its domain and scenario ("adaptive-multi-dweave", 3D "dweave", mesh "dweave")
    std::string title;  // "Adaptive Multi-Object DoubleWeave"
    // The library method behind the entry (2D Enright / 3D Enright; the mesh entry uses neither).
    springls::Method method2D = springls::Method::Springls;
    springls::Method3D method3D = springls::Method3D::Springls;
    // New, uninitialized simulation of this entry for one run (call init() next).
    std::function<std::unique_ptr<Simulation>(const RunOptions &)> make;
};

// The entry's program: <scenario>-<domain>-<key> ("enright-2d-adaptive-multi-dweave", "mesh-3d-dweave").
std::string ProgramName(const Entry &entry);

class Registry
{
public:
    static const Registry &Instance();
    const std::vector<Entry> &entries() const { return mEntries; }
    // The entry implementing config, or null.
    const Entry *find(const Configuration &config) const;
    const Entry *findKey(Domain domain, Scenario scenario, const std::string &key) const;
    // Implemented configurations closest to config (most matching axes first, at most three), from
    // its domain and scenario when it has any, else from all.
    std::vector<const Entry *> nearest(const Configuration &config) const;
    // Markdown table: representation | domain | grid type | objects | scenario | interface tracking |
    // method key | program | title, one row per entry in registry order.
    std::string markdownMatrix() const;

private:
    Registry();
    std::vector<Entry> mEntries;
};
}  // namespace toybox
