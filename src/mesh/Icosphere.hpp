#pragma once
// A sphere mesh: the icosahedron subdivided uniformly and projected onto the sphere.
#include <mesh/LoopSubdivision.hpp>

namespace springls
{
// 20 * 4^levels triangles, counter-clockwise seen from outside.
LoopMesh MakeIcosphere(int levels, double radius = 1.0, const Vec3d &center = Vec3d::Zero());
}  // namespace springls
