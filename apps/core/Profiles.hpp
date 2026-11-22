#pragma once
// Default flag values per (domain, scenario, interface tracking): used for every value neither the
// user nor the demo gives. A selection with any semi-Lagrangian run uses the semi-Lagrangian
// profile for the whole invocation.
#include "Configuration.hpp"
#include "Experiment.hpp"

namespace toybox
{
struct Profile {
    double period = 8.0;
    int maxDepth = 4;  // -1: auto (mesh)
    double maxTimeStepFraction = 1.0 / 800.0;
    int gridSize = 0;  // 0: per method (Enright) / from the mesh (mesh)
    Pacing pacing = Pacing::Lockstep;
    int substepsPerStep = 0;  // 0: the method's default
};

Profile DefaultProfile(Domain domain, Scenario scenario, bool semiLagrangian);
}  // namespace toybox
