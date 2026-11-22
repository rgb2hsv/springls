#pragma once
// The mesh window draws the toybox mesh simulation (apps/core/MeshSimulation.hpp).
#include <core/MeshSimulation.hpp>

namespace viewer
{
using SessionOptions = toybox::MeshOptions;
using SessionStats = toybox::MeshStats;
using SessionSnapshot = toybox::MeshSnapshot;
}  // namespace viewer
