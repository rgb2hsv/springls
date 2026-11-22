#include "Matcap.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
// The implementation is nanovg's (nanovg.cpp).
#include <stb_image.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <filesystem>

namespace viewer
{
namespace fs = std::filesystem;

std::vector<Matcap> LoadMatcaps(std::string &error)
{
    std::vector<Matcap> result;
    std::vector<std::string> candidates = {"assets/matcap"};
#ifdef SPRINGLS_ASSET_DIR
    candidates.push_back(std::string(SPRINGLS_ASSET_DIR) + "/matcap");
#endif
    std::string directory;
    for (const std::string &candidate : candidates) {
        if (fs::is_directory(candidate)) {
            directory = candidate;
            break;
        }
    }
    if (directory.empty()) {
        error = "assets/matcap not found";
        return result;
    }

    std::vector<fs::path> files;
    for (const fs::directory_entry &entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png" && entry.path().stem() != "brush_thumbnail_preview") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const fs::path &file : files) {
        int width = 0, height = 0, channels = 0;
        stbi_uc *data = stbi_load(file.string().c_str(), &width, &height, &channels, 4);
        if (data == nullptr) {
            error += (error.empty() ? "" : "\n") + file.string() + ": " + stbi_failure_reason();
            continue;
        }
        Matcap matcap;
        matcap.name = file.stem().string();
        matcap.width = width;
        matcap.height = height;
        matcap.pixels.assign(data, data + static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
        stbi_image_free(data);
        result.push_back(std::move(matcap));
    }
    return result;
}
}  // namespace viewer
