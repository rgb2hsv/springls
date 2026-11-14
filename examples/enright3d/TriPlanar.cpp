#include "TriPlanar.hpp"

#include "GL.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace enright3d
{
using namespace springls;
namespace
{
// Octree depth colors (as the 2D example's grid colors), finest last.
constexpr unsigned char kDepthColors[][3] = {{70, 70, 82}, {60, 105, 140}, {55, 150, 145}, {200, 165, 60}, {230, 110, 60}, {200, 70, 120}, {150, 80, 200}};
constexpr int kDepthColorCount = static_cast<int>(sizeof(kDepthColors) / sizeof(kDepthColors[0]));
constexpr float kOutlineDarken = 0.55f;
constexpr unsigned char kInterface[3] = {255, 255, 255};
const char *kAxisNames[3] = {"X", "Y", "Z"};

// Domain axes of the image for a slice along `axis`: (right, up).
void ImageAxes(int axis, int &right, int &up)
{
    if (axis == 0) {
        right = 1;
        up = 2;
    } else if (axis == 1) {
        right = 0;
        up = 2;
    } else {
        right = 0;
        up = 1;
    }
}
}  // namespace

TriPlanarView::~TriPlanarView()
{
    for (Plane &plane : mPlanes) {
        if (plane.texture) glDeleteTextures(1, &plane.texture);
    }
}

void TriPlanarView::render(const ExperimentSnapshot &snapshot, const RunSnapshot &run, int axis)
{
    Plane &plane = mPlanes[axis];
    const float D = run.config.voxelScale();
    const int R = kResolution;
    int right, up;
    ImageAxes(axis, right, up);
    mPixels.assign(static_cast<size_t>(R) * R * 4, 255);
    // phi and depth at every pixel center, then colors (block outlines and the interface need neighbors).
    std::vector<float> phi(static_cast<size_t>(R) * R, 1.0f);
    std::vector<int> depth(static_cast<size_t>(R) * R, 0);
    std::vector<int> leaf(static_cast<size_t>(R) * R, -1);
    const float pixel = D / R;
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < R; x++) {
            Vec3 p;
            p[axis] = plane.slice;
            p[right] = (x + 0.5f) * pixel;
            p[up] = (R - 1 - y + 0.5f) * pixel;  // image rows run top to bottom
            const size_t i = static_cast<size_t>(y) * R + x;
            if (run.grid) {
                const int l = run.grid->findLeaf(p);
                const AdaptiveGrid3D::Block &block = run.grid->blocks()[l];
                leaf[i] = l;
                depth[i] = block.key.depth;
                phi[i] = block.interpolate((p - block.origin) / block.h);
            } else if (run.levelSet && !run.levelSet->empty()) {
                phi[i] = run.levelSet->interpolate(p);
            }
        }
    }
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < R; x++) {
            const size_t i = static_cast<size_t>(y) * R + x;
            const unsigned char *base = kDepthColors[std::min(depth[i], kDepthColorCount - 1)];
            float shade = 1.0f;
            for (int k = 0; k < 2; k++) {
                const int nx = x + (k == 0), ny = y + (k == 1);
                if (nx >= R || ny >= R) continue;
                if (leaf[i] != leaf[static_cast<size_t>(ny) * R + nx]) shade = kOutlineDarken;
            }
            unsigned char *out = &mPixels[i * 4];
            for (int c = 0; c < 3; c++) {
                // Inside the surface a little lighter, so the region is readable at coarse depths too.
                const float inside = (phi[i] < 0.0f) ? 1.25f : 1.0f;
                out[c] = static_cast<unsigned char>(std::min(255.0f, base[c] * shade * inside));
            }
        }
    }
    // The surface's cross-section, exactly: every mesh triangle cut by the slice plane, drawn as a
    // line (sampling phi per pixel would miss sheets thinner than a pixel).
    const SurfaceMesh3D &mesh = run.manifold.isoSurface;
    const auto toPixel = [&](const Vec3 &p) { return Eigen::Vector2f(p[right] / pixel - 0.5f, (R - 1) - (p[up] / pixel - 0.5f)); };
    const auto plot = [&](const Eigen::Vector2f &a, const Eigen::Vector2f &b) {
        const int steps = std::max(1, static_cast<int>(std::ceil(2.0f * (b - a).cwiseAbs().maxCoeff())));
        for (int s = 0; s <= steps; s++) {
            const Eigen::Vector2f q = a + (b - a) * (static_cast<float>(s) / steps);
            const int x = static_cast<int>(std::lround(q.x())), y = static_cast<int>(std::lround(q.y()));
            if (x < 0 || y < 0 || x >= R || y >= R) continue;
            unsigned char *out = &mPixels[(static_cast<size_t>(y) * R + x) * 4];
            for (int c = 0; c < 3; c++) out[c] = kInterface[c];
        }
    };
    const int K = mesh.elementSize();
    for (size_t f = 0; f < mesh.faceCount(); f++) {
        const uint32_t *idx = mesh.face(f);
        for (int t = 1; t + 1 < K; t++) {
            const Vec3 v[3] = {mesh.vertexes[idx[0]], mesh.vertexes[idx[t]], mesh.vertexes[idx[t + 1]]};
            const float d[3] = {v[0][axis] - plane.slice, v[1][axis] - plane.slice, v[2][axis] - plane.slice};
            Vec3 cut[2];
            int count = 0;
            for (int e = 0; e < 3 && count < 2; e++) {
                const int a = e, b = (e + 1) % 3;
                if ((d[a] < 0.0f) == (d[b] < 0.0f)) continue;
                cut[count++] = v[a] + (d[a] / (d[a] - d[b])) * (v[b] - v[a]);
            }
            if (count == 2) plot(toPixel(cut[0]), toPixel(cut[1]));
        }
    }
    if (!plane.texture) {
        glGenTextures(1, &plane.texture);
        glBindTexture(GL_TEXTURE_2D, plane.texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_2D, plane.texture);
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, R, R, 0, GL_RGBA, GL_UNSIGNED_BYTE, mPixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    plane.generation = snapshot.generation;
    plane.iteration = run.iteration;
    plane.shownSlice = plane.slice;
    plane.shownRun = &run;
}

void TriPlanarView::draw(const ExperimentSnapshot &snapshot, const RunSnapshot &run)
{
    const float D = run.config.voxelScale();
    // Fit all three slices, their labels and sliders, and the legend without a scrollbar (a
    // scrollbar that comes and goes would resize the images every other frame).
    const ImGuiStyle &style = ImGui::GetStyle();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float perPlane = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.y;
    const float legend = 4.0f * ImGui::GetFrameHeightWithSpacing();  // title + up to three rows of swatches
    const float width = std::max(32.0f, std::min(avail.x, (avail.y - 3.0f * perPlane - legend) / 3.0f));
    for (int axis = 0; axis < 3; axis++) {
        Plane &plane = mPlanes[axis];
        if (plane.slice < 0.0f || plane.slice > D) plane.slice = run.config.center[axis] * D;  // through the initial sphere
        if (!plane.texture || plane.generation != snapshot.generation || plane.iteration != run.iteration || plane.shownSlice != plane.slice ||
            plane.shownRun != &run) {
            render(snapshot, run, axis);
        }
        int right, up;
        ImageAxes(axis, right, up);
        ImGui::Text("%s = %.2f  (%s right, %s up)", kAxisNames[axis], plane.slice, kAxisNames[right], kAxisNames[up]);
        ImGui::Image(static_cast<ImTextureID>(plane.texture), ImVec2(width, width));
        char label[16];
        std::snprintf(label, sizeof(label), "%s slice", kAxisNames[axis]);
        ImGui::SetNextItemWidth(-(ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemInnerSpacing.x));
        ImGui::SliderFloat(label, &plane.slice, 0.0f, D, "%.2f");
    }
    // Depth legend.
    const int maxDepth = run.grid ? run.maxDepth : 0;
    ImGui::TextUnformatted("Octree depth:");
    for (int d = 0; d <= maxDepth; d++) {
        const unsigned char *c = kDepthColors[std::min(d, kDepthColorCount - 1)];
        ImGui::ColorButton("##depth", ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14));
        ImGui::SameLine();
        ImGui::Text("%d (%d^3)", d, run.config.gridSize << d);
        if (d < maxDepth && (d % 2 == 0)) ImGui::SameLine(0.0f, 16.0f);
    }
}
}  // namespace enright3d
