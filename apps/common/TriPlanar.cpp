#include "TriPlanar.hpp"

#include "GL.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace springls_gui
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

void TriPlanarView::render(const TriPlanarSource &source, int axis)
{
    Plane &plane = mPlanes[axis];
    const int R = kResolution;
    int right, up;
    ImageAxes(axis, right, up);
    // Square pixels: the longer image side gets R pixels, the other fewer (a cubic domain gives R x R).
    const float pixel = std::max(source.extent[right], source.extent[up]) / R;
    const int W = std::clamp(static_cast<int>(std::ceil(source.extent[right] / pixel - 1e-4f)), 1, R);
    const int H = std::clamp(static_cast<int>(std::ceil(source.extent[up] / pixel - 1e-4f)), 1, R);
    mPixels.assign(static_cast<size_t>(W) * H * 4, 255);
    // phi and depth at every pixel center, then colors (block outlines and the interface need neighbors).
    std::vector<float> phi(static_cast<size_t>(W) * H, 1.0f);
    std::vector<int> depth(static_cast<size_t>(W) * H, 0);
    std::vector<int> leaf(static_cast<size_t>(W) * H, -1);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            Vec3 p;
            p[axis] = plane.slice;
            p[right] = (x + 0.5f) * pixel;
            p[up] = (H - 1 - y + 0.5f) * pixel;  // image rows run top to bottom
            const size_t i = static_cast<size_t>(y) * W + x;
            if (source.grid) {
                const int l = source.grid->findLeaf(p);
                const AdaptiveGrid3D::Block &block = source.grid->blocks()[l];
                leaf[i] = l;
                depth[i] = block.key.depth;
                phi[i] = block.interpolate((p - block.origin) / block.h);
            } else if (source.levelSet && !source.levelSet->empty()) {
                phi[i] = source.levelSet->interpolate(p);
            }
        }
    }
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const size_t i = static_cast<size_t>(y) * W + x;
            const unsigned char *base = kDepthColors[std::min(depth[i], kDepthColorCount - 1)];
            float shade = 1.0f;
            for (int k = 0; k < 2; k++) {
                const int nx = x + (k == 0), ny = y + (k == 1);
                if (nx >= W || ny >= H) continue;
                if (leaf[i] != leaf[static_cast<size_t>(ny) * W + nx]) shade = kOutlineDarken;
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
    if (source.isoSurface) {
        const SurfaceMesh3D &mesh = *source.isoSurface;
        const auto toPixel = [&](const Vec3 &p) { return Eigen::Vector2f(p[right] / pixel - 0.5f, (H - 1) - (p[up] / pixel - 0.5f)); };
        const auto plot = [&](const Eigen::Vector2f &a, const Eigen::Vector2f &b) {
            const int steps = std::max(1, static_cast<int>(std::ceil(2.0f * (b - a).cwiseAbs().maxCoeff())));
            for (int s = 0; s <= steps; s++) {
                const Eigen::Vector2f q = a + (b - a) * (static_cast<float>(s) / steps);
                const int x = static_cast<int>(std::lround(q.x())), y = static_cast<int>(std::lround(q.y()));
                if (x < 0 || y < 0 || x >= W || y >= H) continue;
                unsigned char *out = &mPixels[(static_cast<size_t>(y) * W + x) * 4];
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, mPixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    plane.shownStamp = source.stamp;
    plane.shownSlice = plane.slice;
    plane.shownIdentity = source.identity;
}

void TriPlanarView::draw(const TriPlanarSource &source)
{
    // Fit all three slices, their labels and sliders, and the legend without a scrollbar (a
    // scrollbar that comes and goes would resize the images every other frame).
    const ImGuiStyle &style = ImGui::GetStyle();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float perPlane = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.y;
    // Title + rows of two depths each (at least three rows, so a shallow grid's layout does not move).
    const int legendDepth = source.grid ? source.maxDepth : 0;
    const float legend = static_cast<float>(1 + std::max(3, (legendDepth + 2) / 2)) * ImGui::GetFrameHeightWithSpacing();
    const float width = std::max(32.0f, std::min(avail.x, (avail.y - 3.0f * perPlane - legend) / 3.0f));
    for (int axis = 0; axis < 3; axis++) {
        Plane &plane = mPlanes[axis];
        if (plane.slice < 0.0f || plane.slice > source.extent[axis]) plane.slice = source.defaultSlice[axis];
        if (!plane.texture || plane.shownStamp != source.stamp || plane.shownSlice != plane.slice || plane.shownIdentity != source.identity) {
            render(source, axis);
        }
        int right, up;
        ImageAxes(axis, right, up);
        ImGui::Text("%s = %.2f  (%s right, %s up)", kAxisNames[axis], plane.slice, kAxisNames[right], kAxisNames[up]);
        // The longer side of the image fills `width`; the shorter one keeps the domain's aspect ratio.
        const float longest = std::max(source.extent[right], source.extent[up]);
        ImGui::Image(static_cast<ImTextureID>(plane.texture), ImVec2(width * source.extent[right] / longest, width * source.extent[up] / longest));
        char label[16];
        std::snprintf(label, sizeof(label), "%s slice", kAxisNames[axis]);
        ImGui::SetNextItemWidth(-(ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemInnerSpacing.x));
        ImGui::SliderFloat(label, &plane.slice, 0.0f, source.extent[axis], "%.2f");
    }
    // Depth legend: the grid resolution at each depth.
    const int maxDepth = source.grid ? source.maxDepth : 0;
    const bool cubic = source.baseCells[0] == source.baseCells[1] && source.baseCells[1] == source.baseCells[2];
    ImGui::TextUnformatted("Octree depth:");
    for (int d = 0; d <= maxDepth; d++) {
        const unsigned char *c = kDepthColors[std::min(d, kDepthColorCount - 1)];
        ImGui::ColorButton("##depth", ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14));
        ImGui::SameLine();
        const Vec3i cells = source.baseCells * (1 << d);
        if (cubic) {
            ImGui::Text("%d (%d^3)", d, cells[0]);
        } else {
            ImGui::Text("%d (%d\xC3\x97%d\xC3\x97%d)", d, cells[0], cells[1], cells[2]);
        }
        if (d < maxDepth && (d % 2 == 0)) ImGui::SameLine(0.0f, 16.0f);
    }
}
}  // namespace springls_gui
