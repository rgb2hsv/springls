#pragma once
// The programs: toybox, and one demo per registry entry named <scenario>-<domain>-<key>
// (ProgramName), which runs that entry: `toybox --domain <domain> --scenario <scenario> -m <key>`.
// A demo's arguments come before the user's, so user flags override them (including -m and the
// axis flags); every other default comes from the profile of what runs.
#include "Registry.hpp"

#include <string>
#include <vector>

namespace toybox
{
struct Demo {
    std::string program;
    std::string description;
    std::vector<std::string> defaultArgs;
};

namespace demos
{
Demo Toybox();                                 // the four 2D CFL-limited methods by default; every axis and flag available
Demo ForEntry(const Entry &entry);             // the entry's program
std::vector<Demo> All();                       // toybox, then one per registry entry in registry order
const Demo *Find(const std::string &program);  // null for an unknown program
}  // namespace demos
}  // namespace toybox
