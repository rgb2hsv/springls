#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace viewer
{
// A matcap ("material capture") image from Blender's studiolights (CC0, see assets/matcap/license.txt):
// diffuse + specular of a white material, sRGB-encoded, 8 bits per channel.
struct Matcap {
    std::string name;  // file stem, e.g. "clay_studio"
    int width = 0, height = 0;
    std::vector<uint8_t> pixels;  // RGBA8, top row first
};

// Loads every .png of assets/matcap (the working directory's, else SPRINGLS_ASSET_DIR/matcap), sorted by
// name. Blender's brush preview ("brush_thumbnail_preview") is skipped. Files that fail to load are
// reported in error and skipped.
std::vector<Matcap> LoadMatcaps(std::string &error);
}  // namespace viewer
