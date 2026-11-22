#include "Profiles.hpp"

namespace toybox
{
Profile DefaultProfile(Domain domain, Scenario scenario, bool semiLagrangian)
{
    Profile p;
    if (scenario == Scenario::Mesh || Surface(scenario)) {  // sphere, stone: no grid, depth or period
        p.maxDepth = -1;
        p.gridSize = 0;
        return p;
    }
    // 2D: enright2d / sl-enright2d; 3D (and, until it has its own, the surface domain): enright3d / sl-enright3d.
    const bool d2 = domain == Domain::D2;
    p.period = d2 ? 8.0 : 3.0;
    if (semiLagrangian) {
        p.maxDepth = 2;
        p.maxTimeStepFraction = 1.0 / 50.0;
        p.gridSize = d2 ? 128 : 32;
        p.pacing = Pacing::EqualCompute;
        p.substepsPerStep = 1;
    } else {
        p.maxDepth = d2 ? 5 : 3;
        // 2D: no time cap, the CFL limit (over each sub-step) alone sets the step; 3D: T/800.
        p.maxTimeStepFraction = d2 ? 0.0 : 1.0 / 800.0;
        p.gridSize = 0;
        p.pacing = Pacing::Lockstep;
        p.substepsPerStep = 0;
    }
    return p;
}
}  // namespace toybox
