#include "Renderer.hpp"

#include "GL.hpp"

#include <springls/AdaptiveSpringLevelSet2D.hpp>

#include "nanovg.hpp"
#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg_gl.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cstdio>
#include <iomanip>
#include <sstream>

namespace enright
{
using namespace springls;
namespace
{
constexpr float kTitleFontSize = 26.0f;
constexpr float kTextFontSize = 18.0f;
constexpr float kTextLineHeight = 22.0f;
constexpr float kReferenceWidth = 1.5f;
// Element sizes are fractions of one level-set cell (scaled by each springl's size class), so
// they scale with zoom; springls, vertexes and particles are at least kMinElementPixels.
constexpr float kContourWidth = 4.0f;  // screen pixels
constexpr float kContourFillAlpha = 0.5f;
constexpr float kSpringlWidthCells = 0.1f;
constexpr float kParticleRadiusCells = 0.1f;
constexpr float kVertexRadiusCells = 0.05f;
constexpr float kNormalWidthCells = 0.05f;
constexpr float kNormalLengthCells = SpringLevelSet2D::kExtent;
constexpr float kCapsuleWidthCells = 2.0f * SpringLevelSet2D::kExtent;
constexpr float kMinVisibleSize = 0.5f;
constexpr float kMinElementPixels = 1.0f;
constexpr int kMetricsPrecision = 5;
// Grid lines: a depth's cell lines are drawn once its cells span this many screen pixels; block
// outlines are always drawn.
constexpr float kMinGridCellPixels = 4.0f;
constexpr float kGridLineWidth = 1.0f;
constexpr int kUniformBlockLines = 8;

nvg::Color Rgb(int r, int g, int b, int a = 255) { return nvg::rgba(r, g, b, a); }
nvg::Color Rgbf(float r, float g, float b, float a = 1.0f) { return nvg::rgbaf(r, g, b, a); }

const nvg::Color kBackground = Rgb(32, 32, 36);
const nvg::Color kPanelColor = Rgb(20, 20, 24);
const nvg::Color kReferenceColor = Rgb(255, 255, 255, 160);
const nvg::Color kContourColor = Rgbf(0.0f, 0.5f, 0.5f);
const nvg::Color kContourFillColor = Rgbf(0.0f, 0.5f, 0.5f, kContourFillAlpha);
const nvg::Color kPrimalSpringlColor = Rgbf(0.5f, 0.5f, 0.5f);
const nvg::Color kPrimalParticleColor = Rgbf(0.6f, 0.0f, 0.0f);
const nvg::Color kDualSpringlColor = Rgbf(1.0f, 0.55f, 0.0f);
const nvg::Color kDualParticleColor = Rgbf(0.0f, 0.45f, 1.0f);
const nvg::Color kVertexColor = Rgbf(1.0f, 0.8f, 0.0f);
const nvg::Color kNormalColor = Rgbf(0.0f, 0.8f, 0.0f, 0.5f);
const nvg::Color kPrimalCapsuleColor = Rgbf(236 / 255.0f, 64 / 255.0f, 122 / 255.0f, 0.5f);
const nvg::Color kDualCapsuleColor = Rgbf(64 / 255.0f, 160 / 255.0f, 236 / 255.0f, 0.5f);
const nvg::Color kGridDepthColors[] = {Rgb(70, 70, 82), Rgb(60, 105, 140), Rgb(55, 150, 145), Rgb(200, 165, 60), Rgb(230, 110, 60)};
constexpr int kGridDepthColorCount = static_cast<int>(sizeof(kGridDepthColors) / sizeof(kGridDepthColors[0]));
const nvg::Color kTextColor = Rgb(230, 230, 230);
const nvg::Color kStatusColor = Rgb(255, 200, 60);
const nvg::Color kTextShadow = Rgb(10, 10, 10);

int Columns(size_t count) { return (count > 1) ? 2 : 1; }
int Rows(size_t count) { return static_cast<int>((count + Columns(count) - 1) / Columns(count)); }

// Level-set pixel -> domain -> shared ROI -> panel, with y pointing up as in the usual Enright plots.
struct PanelTransform {
    float scale;  // pixels per domain unit of the level set
    RegionOfInterest roi;
    Rect panel;
    Vec2 operator()(const Vec2 &p) const
    {
        const Vec2 rel = (p / scale - roi.origin) / roi.size;
        return Vec2(panel.x + rel.x() * panel.w, panel.y + (1.0f - rel.y()) * panel.h);
    }
    float cellPixels() const { return panel.w / roi.size / scale; }  // screen pixels per level-set cell
};

void Shadowed(nvg::Context &vg, float x, float y, const std::string &text, const nvg::Color &color)
{
    vg.fillColor(kTextShadow);
    vg.text(x + 1.0f, y + 1.0f, text.c_str(), nullptr);
    vg.fillColor(color);
    vg.text(x, y, text.c_str(), nullptr);
}
// Metric lines shown under a run's title.
std::vector<std::pair<std::string, nvg::Color>> HeaderLines(const Experiment &experiment, const Run &run, const std::string &status)
{
    const EnrightConfig2D &config = experiment.config();
    const EnrightMetrics2D &mt = run.current;
    const bool hasConstellations = run.sim->getManifold().springlConstellations.size() == run.sim->getManifold().springlCount();
    std::vector<std::pair<std::string, nvg::Color>> lines;
    std::ostringstream s;
    const auto push = [&](const nvg::Color &color) {
        lines.emplace_back(s.str(), color);
        s.str("");
    };
    s << std::fixed << std::setprecision(3) << "t = " << mt.time << " / " << config.period << (run.done ? "  (done)" : status);
    push((!run.done && !status.empty()) ? kStatusColor : kTextColor);
    s << std::fixed << std::setprecision(kMetricsPrecision) << "area error " << 100.0 * mt.relativeAreaError << " %";
    push(kTextColor);
    s << std::fixed << std::setprecision(kMetricsPrecision) << "shape error vs circle: mean " << mt.meanShapeError << "  max " << mt.maxShapeError;
    push(kTextColor);
    const uint64_t substeps = run.sim->getAdvectionSubstepCount();
    s << "advection sub-steps " << substeps << std::fixed << std::setprecision(5) << "   mean dt "
      << (substeps > 0 ? mt.time / static_cast<double>(substeps) : 0.0) << std::setprecision(1) << "   compute " << run.seconds << " s";
    push(kTextColor);
    s << "springls " << mt.springlCount;
    if (hasConstellations) s << "  (primal " << mt.constellationSpringlCounts[0] << ", dual " << mt.constellationSpringlCounts[1] << ")";
    push(kTextColor);
    const int baseSize = config.gridSize;
    if (const auto *adaptive = dynamic_cast<const AdaptiveSpringLevelSet2D *>(run.sim.get())) {
        const AdaptiveGrid2D &grid = adaptive->getGrid();
        int finest = 0;
        for (int index : grid.leaves()) finest = std::max(finest, grid.blocks()[index].key.depth);
        s << "grid " << baseSize << " x " << baseSize << " to " << (baseSize << finest) << " x " << (baseSize << finest) << " equivalent (depths 0-" << finest
          << " of " << grid.maxDepth() << ", " << grid.leaves().size() << " blocks)";
    } else {
        s << "grid " << baseSize << " x " << baseSize << " (uniform)";
    }
    push(kTextColor);
    return lines;
}

}  // namespace

Renderer::Renderer() = default;
Renderer::~Renderer() { shutdown(); }

bool Renderer::init(const std::string &fontFile, std::string &error)
{
    mContext = nvg::createGL(static_cast<int>(nvg::CreateFlags::Antialias | nvg::CreateFlags::StencilStrokes));
    if (!mContext) {
        error = "could not create the NanoVG context";
        return false;
    }
    mFont = mContext->createFont("sans", fontFile.c_str());
    if (mFont < 0) {
        error = "could not load font " + fontFile;
        return false;
    }
    return true;
}

void Renderer::shutdown()
{
    if (mContext) nvg::deleteGL(mContext);
}

std::vector<Rect> Renderer::Layout(size_t count, const Rect &bounds)
{
    std::vector<Rect> panels(count);
    if (count == 0) return panels;
    const int columns = Columns(count), rows = Rows(count);
    const float size =
        std::max(0.0f, std::min((bounds.w - (columns + 1) * kPanelMargin) / columns, (bounds.h - rows * kHeaderHeight - (rows + 1) * kPanelMargin) / rows));
    // Center the block of panels horizontally.
    const float used = columns * size + (columns + 1) * kPanelMargin;
    const float x0 = bounds.x + 0.5f * std::max(0.0f, bounds.w - used);
    for (size_t i = 0; i < count; i++) {
        const int column = static_cast<int>(i) % columns, row = static_cast<int>(i) / columns;
        panels[i] = {x0 + kPanelMargin + column * (size + kPanelMargin), bounds.y + kPanelMargin + kHeaderHeight + row * (size + kHeaderHeight + kPanelMargin),
                     size, size};
    }
    return panels;
}

void Renderer::ImageSize(size_t count, float panelSize, int &width, int &height)
{
    const int columns = Columns(count), rows = Rows(count);
    width = static_cast<int>(std::ceil(columns * panelSize + (columns + 1) * kPanelMargin));
    height = static_cast<int>(std::ceil(rows * (panelSize + kHeaderHeight) + (rows + 1) * kPanelMargin));
    width += width & 1;
    height += height & 1;
}

void Renderer::draw(const Experiment &experiment, const Visibility &visibility, const RegionOfInterest &roi, const Rect &bounds, float viewWidth,
                    float viewHeight, float devicePixelRatio, const std::string &status)
{
    nvg::Context &vg = *mContext;
    mPanels = Layout(experiment.runs().size(), bounds);
    vg.beginFrame(viewWidth, viewHeight, devicePixelRatio);
    vg.fillColor(kBackground);
    vg.beginPath();
    vg.rect(bounds.x, bounds.y, bounds.w, bounds.h);
    vg.fill();
    vg.fontFaceId(mFont);
    vg.textAlign(static_cast<int>(nvg::Align::Left | nvg::Align::Top));
    vg.lineCap(static_cast<int>(nvg::LineCap::Round));
    vg.lineJoin(static_cast<int>(nvg::LineCap::Round));
    // One header text scale for every panel, so the widest line fits its panel.
    float headerScale = 1.0f;
    vg.fontSize(kTextFontSize);
    for (size_t i = 0; i < mPanels.size(); i++) {
        for (const auto &line : HeaderLines(experiment, experiment.runs()[i], status)) {
            const float width = vg.textBounds(0.0f, 0.0f, line.first.c_str(), nullptr, nullptr);
            if (width > mPanels[i].w && width > 0.0f) headerScale = std::min(headerScale, mPanels[i].w / width);
        }
    }
    for (size_t i = 0; i < mPanels.size(); i++) drawRun(experiment, experiment.runs()[i], visibility, roi, mPanels[i], status, headerScale);
    vg.endFrame();
}

void Renderer::drawRun(const Experiment &experiment, const Run &run, const Visibility &visibility, const RegionOfInterest &roi, const Rect &panel,
                       const std::string &status, float headerScale)
{
    nvg::Context &vg = *mContext;
    const EnrightConfig2D &config = experiment.config();
    const SpringLevelSet2D &sim = *run.sim;
    const Manifold2D &m = sim.getManifold();
    const PanelTransform toScreen{config.pixelScale(), roi, panel};
    const float pixelsPerDomainUnit = panel.w / roi.size;
    const float cellSize = toScreen.cellPixels();

    vg.save();
    vg.fillColor(kPanelColor);
    vg.beginPath();
    vg.rect(panel.x, panel.y, panel.w, panel.h);
    vg.fill();
    vg.scissor(panel.x, panel.y, panel.w, panel.h);

    if (visibility.grid) drawGrid(experiment, run, roi, panel);
    if (visibility.referenceCircle) {
        // Exact solution at t = 0 and t = T.
        const Vec2 c = toScreen(config.centerPixels());
        vg.strokeColor(kReferenceColor);
        vg.strokeWidth(kReferenceWidth);
        vg.beginPath();
        vg.circle(c.x(), c.y(), config.radius * pixelsPerDomainUnit);
        vg.stroke();
    }
    if (visibility.contour || visibility.contourFill) {
        vg.fillColor(kContourFillColor);
        vg.strokeColor(kContourColor);
        vg.strokeWidth(kContourWidth);
        for (const Curve &curve : m.vertexIndexes) {
            vg.beginPath();
            for (size_t k = 0; k < curve.size(); k++) {
                const Vec2 p = toScreen(m.vertexLocations[curve[k]]);
                if (k == 0) vg.moveTo(p.x(), p.y());
                else vg.lineTo(p.x(), p.y());
            }
            vg.closePath();
            if (visibility.contourFill) vg.fill();
            if (visibility.contour) vg.stroke();
        }
    }

    const size_t SN = m.springlParticles.size();
    const bool hasConstellations = m.springlConstellations.size() == SN;
    const auto isDual = [&](size_t n) { return hasConstellations && m.springlConstellations[n] == static_cast<uint8_t>(Constellation::Dual); };
    const auto isVisible = [&](size_t n) { return isDual(n) ? visibility.dual : visibility.primal; };
    const auto elementSize = [&](size_t n, float cells) { return cells * cellSize * sim.springlScale(n); };
    const auto clampedSize = [&](size_t n, float cells) { return std::max(kMinElementPixels, elementSize(n, cells)); };
    // Springl polyline vertex 0 -> particle -> vertex 1.
    const auto strokeSpringls = [&](float widthCells, const nvg::Color &primal, const nvg::Color &dual) {
        for (size_t n = 0; n < SN; n++) {
            if (!isVisible(n)) continue;
            const Vec2 a = toScreen(m.springlVertexes[2 * n]), p = toScreen(m.springlParticles[n]), b = toScreen(m.springlVertexes[2 * n + 1]);
            vg.strokeWidth(clampedSize(n, widthCells));
            vg.strokeColor(isDual(n) ? dual : primal);
            vg.beginPath();
            vg.moveTo(a.x(), a.y());
            vg.lineTo(p.x(), p.y());
            vg.lineTo(b.x(), b.y());
            vg.stroke();
        }
    };
    if (visibility.capsules) strokeSpringls(kCapsuleWidthCells, kPrimalCapsuleColor, kDualCapsuleColor);
    if (visibility.springls) strokeSpringls(kSpringlWidthCells, kPrimalSpringlColor, kDualSpringlColor);
    if (visibility.normals && m.springlParticleNormals.size() == SN) {
        vg.strokeColor(kNormalColor);
        for (size_t n = 0; n < SN; n++) {
            if (!isVisible(n) || elementSize(n, kNormalWidthCells) <= kMinVisibleSize) continue;
            const Vec2 p = toScreen(m.springlParticles[n]);
            const Vec2 q = toScreen(m.springlParticles[n] + kNormalLengthCells * sim.springlScale(n) * m.springlParticleNormals[n]);
            vg.strokeWidth(elementSize(n, kNormalWidthCells));
            vg.beginPath();
            vg.moveTo(p.x(), p.y());
            vg.lineTo(q.x(), q.y());
            vg.stroke();
        }
    }
    if (visibility.vertexes) {
        vg.fillColor(kVertexColor);
        for (size_t n = 0; n < 2 * SN; n++) {
            if (!isVisible(n / 2)) continue;
            const Vec2 v = toScreen(m.springlVertexes[n]);
            vg.beginPath();
            vg.circle(v.x(), v.y(), clampedSize(n / 2, kVertexRadiusCells));
            vg.fill();
        }
    }
    if (visibility.particles) {
        for (size_t n = 0; n < SN; n++) {
            if (!isVisible(n)) continue;
            const Vec2 p = toScreen(m.springlParticles[n]);
            vg.fillColor(isDual(n) ? kDualParticleColor : kPrimalParticleColor);
            vg.beginPath();
            vg.circle(p.x(), p.y(), clampedSize(n, kParticleRadiusCells));
            vg.fill();
        }
    }
    vg.restore();

    // Header above the panel.
    float x = panel.x, y = panel.y - kHeaderHeight;
    vg.fontSize(kTitleFontSize * headerScale);
    Shadowed(vg, x, y, MethodTitle(run.method), kTextColor);
    y += (kTextLineHeight + kTitleFontSize - kTextFontSize) * headerScale;
    vg.fontSize(kTextFontSize * headerScale);
    for (const auto &[text, color] : HeaderLines(experiment, run, status)) {
        Shadowed(vg, x, y, text, color);
        y += kTextLineHeight * headerScale;
    }
}

void Renderer::drawGrid(const Experiment &experiment, const Run &run, const RegionOfInterest &roi, const Rect &panel)
{
    nvg::Context &vg = *mContext;
    const EnrightConfig2D &config = experiment.config();
    const float scale = config.pixelScale();
    const PanelTransform toScreen{scale, roi, panel};
    const float cellSize = toScreen.cellPixels();
    // Visible region in level-set pixels, for culling.
    const Vec2 visibleMin = roi.origin * scale;
    const Vec2 visibleMax = (roi.origin + Vec2::Constant(roi.size)) * scale;
    const auto addLine = [&](const Vec2 &a, const Vec2 &b) {
        const Vec2 p = toScreen(a), q = toScreen(b);
        vg.moveTo(p.x(), p.y());
        vg.lineTo(q.x(), q.y());
    };
    vg.strokeWidth(kGridLineWidth);
    if (const auto *adaptive = dynamic_cast<const AdaptiveSpringLevelSet2D *>(run.sim.get())) {
        const AdaptiveGrid2D &grid = adaptive->getGrid();
        constexpr int N = AdaptiveGrid2D::kBlockCells;
        // Every depth draws one line per `stride` of its own cells, with one stride for all depths,
        // chosen so the finest depth's lines are kMinGridCellPixels apart: each depth is then drawn
        // twice as densely as the one above it, as it really is, at any zoom. Strides past the block
        // size draw block outlines only.
        int finest = 0;
        for (int index : grid.leaves()) finest = std::max(finest, grid.blocks()[index].key.depth);
        const float finestPixels = cellSize * std::ldexp(1.0f, -finest);
        int stride = 1;
        while (stride * finestPixels < kMinGridCellPixels && stride < N) stride *= 2;
        for (int depth = 0; depth <= grid.maxDepth(); depth++) {
            vg.beginPath();
            bool any = false;
            for (int index : grid.leaves()) {
                const AdaptiveGrid2D::Block &block = grid.blocks()[index];
                if (block.key.depth != depth) continue;
                const Vec2 lo = block.origin, hi = block.origin + Vec2::Constant(block.size());
                if (hi.x() < visibleMin.x() || hi.y() < visibleMin.y() || lo.x() > visibleMax.x() || lo.y() > visibleMax.y()) continue;
                const int step = stride;
                for (int k = 0; k <= N; k += step) {
                    const float t = k * block.h;
                    addLine(lo + Vec2(t, 0.0f), Vec2(lo.x() + t, hi.y()));
                    addLine(lo + Vec2(0.0f, t), Vec2(hi.x(), lo.y() + t));
                }
                any = true;
            }
            if (any) {
                vg.strokeColor(kGridDepthColors[std::min(depth, kGridDepthColorCount - 1)]);
                vg.stroke();
            }
        }
    } else {
        // Uniform grid: every cell when large enough on screen, otherwise every kUniformBlockLines cells.
        const int last = config.gridSize - 1;
        const int step = (cellSize >= kMinGridCellPixels) ? 1 : kUniformBlockLines;
        vg.beginPath();
        for (int k = 0; k <= last; k += step) {
            const float t = static_cast<float>(k);
            if (t >= visibleMin.x() - 1 && t <= visibleMax.x() + 1) addLine(Vec2(t, 0.0f), Vec2(t, static_cast<float>(last)));
            if (t >= visibleMin.y() - 1 && t <= visibleMax.y() + 1) addLine(Vec2(0.0f, t), Vec2(static_cast<float>(last), t));
        }
        vg.strokeColor(kGridDepthColors[0]);
        vg.stroke();
    }
}

// -------------------------------------------------------------------------------------------------
// FrameRecorder
// -------------------------------------------------------------------------------------------------

FrameRecorder::~FrameRecorder() { release(); }

void FrameRecorder::release()
{
    if (mFramebuffer) glDeleteFramebuffers(1, &mFramebuffer);
    if (mColor) glDeleteRenderbuffers(1, &mColor);
    if (mDepthStencil) glDeleteRenderbuffers(1, &mDepthStencil);
    mFramebuffer = mColor = mDepthStencil = 0;
}

std::filesystem::path FrameRecorder::UnusedDirectory(const std::filesystem::path &requested)
{
    namespace fs = std::filesystem;
    const auto isFree = [](const fs::path &dir) {
        std::error_code ec;
        if (!fs::exists(dir, ec)) return true;
        return fs::is_directory(dir, ec) && fs::is_empty(dir, ec);
    };
    if (isFree(requested)) return requested;
    // requested/ -> requested_001, requested_002, ...
    fs::path base = requested;
    if (!base.has_filename()) base = base.parent_path();  // trailing separator
    for (int n = 1;; n++) {
        char suffix[16];
        std::snprintf(suffix, sizeof(suffix), "_%03d", n);
        const fs::path candidate = base.parent_path() / (base.filename().string() + suffix);
        if (isFree(candidate)) return candidate;
    }
}

bool FrameRecorder::init(const std::filesystem::path &directory, int width, int height, std::string &error)
{
    release();
    // Never overwrite earlier frames: a directory that already has files is left alone.
    mDirectory = UnusedDirectory(directory);
    mWidth = width;
    mHeight = height;
    mFrame = 0;
    std::error_code ec;
    std::filesystem::create_directories(mDirectory, ec);
    if (ec) {
        error = "could not create " + mDirectory.string() + ": " + ec.message();
        return false;
    }
    glGenFramebuffers(1, &mFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer);
    glGenRenderbuffers(1, &mColor);
    glBindRenderbuffer(GL_RENDERBUFFER, mColor);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mColor);
    glGenRenderbuffers(1, &mDepthStencil);
    glBindRenderbuffer(GL_RENDERBUFFER, mDepthStencil);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, mDepthStencil);
    const bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    if (!complete) {
        release();
        error = "offscreen framebuffer is incomplete";
        return false;
    }
    mPixels.resize(static_cast<size_t>(width) * height * 4);
    return true;
}

bool FrameRecorder::capture(Renderer &renderer, const Experiment &experiment, const Visibility &visibility, const RegionOfInterest &roi)
{
    if (!active()) return false;
    GLint previous = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previous);
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer);
    glViewport(0, 0, mWidth, mHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    const float w = static_cast<float>(mWidth), h = static_cast<float>(mHeight);
    renderer.draw(experiment, visibility, roi, Rect{0.0f, 0.0f, w, h}, w, h, 1.0f, "");
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, mPixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previous));
    // Opaque, top row first.
    const size_t row = static_cast<size_t>(mWidth) * 4;
    for (size_t i = 3; i < mPixels.size(); i += 4) mPixels[i] = 255;
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%06llu.png", static_cast<unsigned long long>(mFrame));
    const std::string file = (mDirectory / name).string();
    stbi_flip_vertically_on_write(1);
    const bool ok = stbi_write_png(file.c_str(), mWidth, mHeight, 4, mPixels.data(), static_cast<int>(row)) != 0;
    if (ok) mFrame++;
    return ok;
}
}  // namespace enright
