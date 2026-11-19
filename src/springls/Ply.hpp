#pragma once
#include <springls/IsoSurface3D.hpp>
#include <springls/Manifold3D.hpp>

#include <filesystem>
#include <string>

namespace springls
{
// PLY meshes, read and written with happly. Every function returns false (with error) on failure.

// Iso-surface mesh: vertex x, y, z (float) and nx, ny, nz when the mesh has one normal per vertex; face
// vertex_indices (uint32). Reading accepts any PLY whose faces are all triangles or all quads; normals are read
// when present and recomputed (SurfaceMesh3D::updateVertexNormals) otherwise.
bool WriteMeshPly(const std::filesystem::path &file, const SurfaceMesh3D &mesh, bool binary, std::string &error);
bool ReadMeshPly(const std::filesystem::path &file, SurfaceMesh3D &mesh, std::string &error);

// Springls of a manifold, one polygon face each: vertex x, y, z are springlVertexes and face n has
// vertex_indices springlBegin(n) .. springlBegin(n + 1) - 1 plus its particle (px, py, pz) and normal (nx, ny, nz)
// as face properties, so viewers show the springl elements and the file round-trips the springl geometry.
// Reading replaces the manifold's springls (particles, vertexes, offsets and normals; the iso-surface, element
// type and other per-springl attributes are left as they are). Faces without particles get their centroid,
// and without normals the fan normal (Manifold3D::updateNormals).
bool WriteSpringlsPly(const std::filesystem::path &file, const Manifold3D &manifold, bool binary, std::string &error);
bool ReadSpringlsPly(const std::filesystem::path &file, Manifold3D &manifold, std::string &error);
}  // namespace springls
