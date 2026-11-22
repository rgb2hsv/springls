#include "Gui.hpp"
#include "../common/Palette.hpp"

#include "GL.hpp"
#include "Matcap.hpp"
#include "Offscreen.hpp"
#include "OrbitCamera.hpp"
#include "TriPlanar.hpp"
#include "ViewRenderer.hpp"

#include <core/Common.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace viewer
{
using namespace springls;
namespace
{
constexpr float kSidebarWidth = 300.0f;
constexpr float kSliceSidebarWidth = 300.0f;  // right sidebar: tri-planar slices of the grid
constexpr int kScreenshotWidth = 1600, kScreenshotHeight = 900;
constexpr int kScreenshotSettleFrames = 3;             // ImGui lays out new windows over the first frames
constexpr const char *kDefaultMatcap = "clay_studio";  // meshlets' default
constexpr const char *kWaitingText = "Waiting for the current step to finish...";
constexpr const char *kInitializingText = "Initializing...";
const ImVec4 kWarningColor(1.0f, 0.8f, 0.3f, 1.0f);
const ImVec4 kErrorColor(1.0f, 0.4f, 0.35f, 1.0f);

void GlfwError(int code, const char *description) { std::cerr << "GLFW error " << code << ": " << description << std::endl; }

// Creates an OpenGL 3.3 core window (hidden when !visible) and makes it current.
GLFWwindow *CreateWindow(int width, int height, bool visible, const std::string &title, std::string &error)
{
    glfwSetErrorCallback(GlfwError);
    if (!glfwInit()) {
        error = "could not initialize GLFW (is a display available?)";
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_SAMPLES, visible ? 4 : 0);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        error = "could not create an OpenGL 3.3 window";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    if (!toybox::LoadGL()) {
        error = "could not load the OpenGL 3.3 entry points";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }
    return window;
}

// Owns the mesh simulation and runs its load (init) and steps on a thread of its own: an init or step takes
// seconds to minutes and cannot be interrupted, so the UI never waits on it. The UI posts requests
// (load, step, play, resampling, quit) and picks up the snapshot the worker copies out after every
// load and step; it never touches the simulation. A load request takes priority over steps and clears
// queued ones, but it waits for a running step to finish. The destructor joins the thread before the
// Session is destroyed.
class SessionWorker
{
public:
    struct Status {
        bool busy = false;        // a load or step is running
        bool loading = false;     // the running job is a load
        bool loadQueued = false;  // a load waits (for the running step)
        bool exited = false;      // the thread returned (after requestQuit)
        bool playing = false;
        bool stepFailed = false;  // a step threw: Play and Step do nothing until the next successful load
        int pendingSteps = 0;
        std::string error;  // of the last load or step; empty when it succeeded
    };

    SessionWorker(const toybox::Entry &entry, bool playing, bool resampling, int threads)
        : mEntry(entry), mPlaying(playing), mResampling(resampling), mThreads(threads), mThread([this] { run(); })
    {}
    ~SessionWorker()
    {
        requestQuit();
        mThread.join();
    }
    SessionWorker(const SessionWorker &) = delete;
    SessionWorker &operator=(const SessionWorker &) = delete;

    // Replaces the session by one loaded with `options` (after the running step, if any).
    void requestLoad(const SessionOptions &options)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mLoadRequest = options;
            mPendingSteps = 0;
        }
        mChanged.notify_all();
    }
    void requestStep()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (!mStepFailed) mPendingSteps++;
        }
        mChanged.notify_all();
    }
    void setPlaying(bool playing)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mPlaying = playing && !mStepFailed;
        }
        mChanged.notify_all();
    }
    // Applied before every step (and after every load).
    void setResampling(bool on)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mResampling = on;
    }
    // The thread returns once the running job (if any) finishes; status().exited tells when.
    void requestQuit()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mQuit = true;
        }
        mChanged.notify_all();
    }
    Status status() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        Status s;
        s.busy = mBusy;
        s.loading = mLoading;
        s.loadQueued = mLoadRequest.has_value();
        s.exited = mExited;
        s.playing = mPlaying;
        s.stepFailed = mStepFailed;
        s.pendingSteps = mPendingSteps;
        s.error = mError;
        return s;
    }
    // Sets `out` to the newest snapshot if one was published since the last call, and `load` to the number of
    // loads finished before it (a change means a new load, even when later steps' snapshots replaced its own).
    // `mesh` is the file the snapshot was loaded from.
    bool fetch(std::shared_ptr<const SessionSnapshot> &out, uint64_t &load, std::filesystem::path &mesh)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mFetched == mPublished) return false;
        out = mLatest;
        load = mLatestLoad;
        mesh = mLatestMesh;
        mFetched = mPublished;
        return true;
    }
    // What the snapshots belong to (for TriPlanarSource::identity); never dereferenced by the UI.
    const void *identity() const { return this; }

private:
    const toybox::Entry &mEntry;                       // makes every simulation
    std::unique_ptr<toybox::MeshSimulation> mSession;  // only the worker thread touches it
    mutable std::mutex mMutex;
    std::condition_variable mChanged;
    bool mPlaying, mResampling;
    int mThreads;
    bool mQuit = false, mBusy = false, mLoading = false, mExited = false;
    int mPendingSteps = 0;
    std::optional<SessionOptions> mLoadRequest;
    std::string mError;
    bool mStepFailed = false;  // a step threw: no more steps until the next successful load
    std::shared_ptr<const SessionSnapshot> mLatest;
    std::filesystem::path mLoadedMesh, mLatestMesh;
    uint64_t mPublished = 0, mFetched = 0, mLoads = 0, mLatestLoad = 0;
    std::thread mThread;  // last, so it starts after every other member is initialized

    void run()
    {
        // OpenMP's thread count is per thread: set it here, on the thread that runs every init and step.
        SetThreads(mThreads);
        std::unique_lock<std::mutex> lock(mMutex);
        while (true) {
            mChanged.wait(lock,
                          [this] { return mQuit || mLoadRequest || ((mPlaying || mPendingSteps > 0) && mSession && mSession->loaded() && !mStepFailed); });
            if (mQuit) break;
            std::optional<SessionOptions> load;
            if (mLoadRequest) {
                load = std::move(mLoadRequest);
                mLoadRequest.reset();
                mLoading = true;
            } else if (mPendingSteps > 0) {
                mPendingSteps--;
            }
            const bool resampling = mResampling;
            mBusy = true;
            lock.unlock();

            std::string error;
            std::shared_ptr<const SessionSnapshot> snapshot;
            bool loaded = false;  // the load replaced the session
            try {
                if (load) {
                    // A new simulation from the registry; a failed load keeps the current one.
                    toybox::RunOptions run;
                    run.mesh = *load;
                    run.elementType = load->elementType;
                    run.tuning = load->tuning;
                    std::unique_ptr<toybox::Simulation> made = mEntry.make(run);
                    auto *fresh = dynamic_cast<toybox::MeshSimulation *>(made.get());
                    if (fresh && fresh->init(error)) {
                        made.release();
                        mSession.reset(fresh);
                        loaded = true;
                        mSession->setResampling(resampling);
                    } else {
                        std::cerr << "error: " << error << std::endl;
                    }
                } else {
                    mSession->setResampling(resampling);
                    mSession->step();
                }
                if (mSession) snapshot = std::make_shared<const SessionSnapshot>(mSession->snapshot());
            } catch (const std::exception &e) {
                error = std::string(load ? "load" : "step") + " failed: " + e.what();
                std::cerr << "error: " << error << std::endl;
            }

            lock.lock();
            mBusy = mLoading = false;
            if (load) {
                mError = error;
                if (loaded) {
                    // The new session replaces the old one (and its failed step).
                    if (mStepFailed) mPlaying = false;
                    mStepFailed = false;
                    mLoadedMesh = load->mesh;
                }
            } else if (!error.empty()) {
                // The session may be inconsistent after a step threw: keep the error and refuse steps until a load.
                mError = error;
                mStepFailed = true;
            }
            if (!error.empty()) {
                mPlaying = false;
                mPendingSteps = 0;
            }
            if (loaded) mLoads++;
            // A failed load keeps the previous session and its snapshot: nothing new to show.
            if (snapshot && (!load || loaded)) {
                mLatestLoad = mLoads;
                mLatestMesh = mLoadedMesh;
                mLatest = std::move(snapshot);
                mPublished++;
            }
        }
        mExited = true;
    }
};

// Grid-space bounds of the input mesh.
void MeshBounds(const SurfaceMesh3D &mesh, Vec3 &lo, Vec3 &hi)
{
    lo = hi = Vec3::Zero();
    if (mesh.vertexes.empty()) return;
    lo = hi = mesh.vertexes[0];
    for (const Vec3 &p : mesh.vertexes) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
}

// The UI's copy of the grid settings; they take effect on the next Load or Reset.
struct GridControls {
    float cellScale = 1.0f;
    int gridOverride = 0;
    bool autoDepth = true;
    int maxDepth = 4;
    int element = 0;  // 0 quads, 1 triangles

    explicit GridControls(const SessionOptions &options)
        : cellScale(options.grid.cellScale),
          gridOverride(options.grid.gridOverride),
          autoDepth(options.grid.maxDepth < 0),
          maxDepth(options.grid.maxDepth < 0 ? 4 : options.grid.maxDepth),
          element(options.elementType == ElementType::Quad ? 0 : 1)
    {}
    void apply(SessionOptions &options) const
    {
        options.grid.cellScale = std::max(cellScale, 1e-3f);
        options.grid.gridOverride = std::max(gridOverride, 0);
        options.grid.maxDepth = autoDepth ? -1 : maxDepth;
        options.elementType = (element == 0) ? ElementType::Quad : ElementType::Triangle;
    }
};

// Right-aligned label slider, as enright3d's.
bool LabeledSlider(const char *label, float *value, float lo, float hi, const char *format = "%.2f")
{
    ImGui::SetNextItemWidth(-(ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemInnerSpacing.x));
    return ImGui::SliderFloat(label, value, lo, hi, format);
}

// A checkbox with a color swatch at the right edge of the sidebar.
void LayerToggle(const char *label, bool *on, std::array<float, 3> *color)
{
    ImGui::Checkbox(label, on);
    if (color) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight());
        ImGui::PushID(label);
        ImGui::ColorEdit3("##color", color->data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        ImGui::PopID();
    }
}

// Thumbnail picker, wrapping as meshlets'.
void MatcapPicker(const ViewRenderer &renderer, const std::vector<Matcap> &matcaps, int &selected)
{
    constexpr float kThumb = 26.0f;
    const ImGuiStyle &style = ImGui::GetStyle();
    const float rowWidth = ImGui::GetContentRegionAvail().x;
    const float buttonWidth = kThumb + style.FramePadding.x * 2.0f;
    float x = 0.0f;
    for (size_t i = 0; i < matcaps.size(); i++) {
        if (i > 0 && x + style.ItemSpacing.x + buttonWidth <= rowWidth) {
            ImGui::SameLine();
            x += style.ItemSpacing.x;
        } else {
            x = 0.0f;
        }
        x += buttonWidth;
        const bool isSelected = selected == static_cast<int>(i);
        ImGui::PushID(static_cast<int>(i));
        const ImVec4 background = isSelected ? ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive) : ImVec4(0, 0, 0, 0);
        if (ImGui::ImageButton("##matcap", static_cast<ImTextureID>(renderer.matcapTexture(static_cast<int>(i))), ImVec2(kThumb, kThumb), ImVec2(0, 0),
                               ImVec2(1, 1), background)) {
            selected = static_cast<int>(i);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", matcaps[i].name.c_str());
        ImGui::PopID();
    }
}

void StatsPanel(const SessionSnapshot &snap)
{
    const SessionStats &s = snap.stats;
    ImGui::Text("steps: %llu  (last %.2f s)", static_cast<unsigned long long>(s.steps), s.lastStepSeconds);
    ImGui::Text("init: %.2f s  (%d threads%s)", s.initSeconds, s.threads, s.fromCache ? ", cached" : "");
    if (!s.fromCache) ImGui::Text("settled in %d iterations", s.settleIterations);
    ImGui::Text("springls: %zu primal, %zu dual", s.primal, s.dual);
    ImGui::Text("iso-surface: %zu verts, %zu faces", s.isoVertexes, s.isoFaces);
    ImGui::Text("octree: %zu leaves, depth %d", s.leaves, s.finestDepth);
    ImGui::Text("base cells: %d x %d x %d", s.baseCells.x(), s.baseCells.y(), s.baseCells.z());
    ImGui::Text("finest: %d cells, h = %.4g", s.baseCells.maxCoeff() << s.finestDepth, s.finestCell);
    ImGui::Text("input: %zu faces (%zu triangles)", s.inputFaces, snap.inputMesh.faceCount());
}
}  // namespace

int RunViewerGui(const toybox::Entry &entry, const SessionOptions &options, const std::string &initialMatcap, bool startPaused, bool zUp,
                 const ScreenshotOptions &screenshot)
{
    const bool capturing = !screenshot.path.empty();
    std::string error;
    int width = kScreenshotWidth, height = kScreenshotHeight;
    if (!capturing) {
        // Fit the window to the monitor work area.
        width = 1700;
        height = 1100;
        if (glfwInit()) {
            int x, y, w, h;
            if (GLFWmonitor *monitor = glfwGetPrimaryMonitor()) {
                glfwGetMonitorWorkarea(monitor, &x, &y, &w, &h);
                if (w > 0 && h > 0) {
                    width = std::min(width, static_cast<int>(0.95f * w));
                    height = std::min(height, static_cast<int>(0.9f * h));
                }
            }
        }
    }
    GLFWwindow *window = CreateWindow(width, height, !capturing, "Springls Viewer", error);
    if (!window) {
        std::cerr << "error: " << error << " (use --no-gui to run without a window)" << std::endl;
        return 1;
    }
    glfwSwapInterval(capturing ? 0 : 1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    int status = 0;
    {
        std::string matcapError;
        const std::vector<Matcap> matcaps = LoadMatcaps(matcapError);
        if (!matcapError.empty()) std::cerr << "warning: " << matcapError << std::endl;
        Layers layers;
        const std::string matcapName = initialMatcap.empty() ? kDefaultMatcap : initialMatcap;
        const auto found = std::find_if(matcaps.begin(), matcaps.end(), [&](const Matcap &m) { return m.name == matcapName; });
        if (found != matcaps.end()) {
            layers.matcap = static_cast<int>(found - matcaps.begin());
        } else if (!matcaps.empty()) {
            std::cerr << "warning: no matcap named '" << matcapName << "'; available:";
            for (const Matcap &m : matcaps) std::cerr << " " << m.name;
            std::cerr << std::endl;
        }

        ViewRenderer renderer;
        springls_gui::OffscreenTarget target;
        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if (!renderer.init(matcaps, error)) {
            std::cerr << "error: " << error << std::endl;
            status = 1;
        } else if (capturing && !target.init(fbW, fbH)) {
            std::cerr << "error: could not create a " << fbW << " x " << fbH << " offscreen framebuffer" << std::endl;
            status = 1;
        }

        OrbitCamera camera;
        camera.zUp = zUp;
        auto slices = std::make_unique<springls_gui::TriPlanarView>();
        bool playing = !startPaused && !capturing && screenshot.steps == 0;  // --steps N: run N steps, then wait
        bool resampling = true;
        bool quitting = false;
        bool fitPending = true;  // fit the camera to the next loaded mesh
        Vec3 fittedLo = Vec3::Zero(), fittedHi = Vec3::Zero();
        int settleFrames = 0;
        SessionOptions requested = options;  // what Reset reloads
        GridControls grid(options);
        char meshPath[1024];
        std::snprintf(meshPath, sizeof(meshPath), "%s", options.mesh.string().c_str());
        std::shared_ptr<const SessionSnapshot> shown;  // the newest snapshot from the worker
        uint64_t shownLoad = 0, framedLoad = 0;        // loads finished before `shown`; the last one framed
        std::filesystem::path shownMesh;               // the file `shown` was loaded from

        // Not started when the GL setup failed (it would load the mesh only to be joined).
        // Declared after everything it shares a frame with, so it is joined first when the scope ends.
        std::unique_ptr<SessionWorker> worker;
        if (status == 0) worker = std::make_unique<SessionWorker>(entry, playing, resampling, options.threads);
        if (status == 0) worker->requestLoad(requested);
        if (status == 0) {
            for (int i = 0; i < screenshot.steps; i++) worker->requestStep();
        }
        const auto load = [&](const std::filesystem::path &mesh) {
            requested.mesh = mesh;
            grid.apply(requested);
            worker->requestLoad(requested);
        };
        const auto resetCamera = [&]() {
            camera = OrbitCamera();
            camera.zUp = zUp;
            camera.fit(fittedLo, fittedHi);
        };

        while (status == 0) {
            if (capturing) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            glfwPollEvents();
            if (glfwWindowShouldClose(window) && !quitting) {
                // The running init or step cannot be interrupted: keep drawing until the worker returns.
                quitting = true;
                worker->requestQuit();
            }
            if (worker->fetch(shown, shownLoad, shownMesh) && shownLoad != framedLoad && shown && shown->grid) {
                // A new load: frame it (on Load, or when Reset changed the grid space).
                framedLoad = shownLoad;
                // Show the depth in use, so unchecking auto and resetting keeps it.
                if (grid.autoDepth) grid.maxDepth = shown->stats.finestDepth;
                Vec3 lo, hi;
                MeshBounds(shown->inputMesh, lo, hi);
                if (fitPending || !lo.isApprox(fittedLo) || !hi.isApprox(fittedHi)) {
                    fittedLo = lo;
                    fittedHi = hi;
                    camera.fit(lo, hi);
                    slices = std::make_unique<springls_gui::TriPlanarView>();  // slices back to the center
                    fitPending = false;
                }
            }
            const SessionWorker::Status state = worker->status();
            if (quitting && state.exited) break;
            playing = state.playing;  // the worker pauses itself after an error

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            int winW, winH;
            glfwGetWindowSize(window, &winW, &winH);
            glfwGetFramebufferSize(window, &fbW, &fbH);
            const float pixelRatio = (winW > 0) ? static_cast<float>(fbW) / winW : 1.0f;
            const bool hasMesh = shown && shown->grid;

            // What the worker is doing, for the sidebar and the viewport overlay.
            const char *activity = nullptr;
            if (quitting || (state.loadQueued && state.busy && !state.loading)) {
                activity = kWaitingText;
            } else if (state.loading || state.loadQueued) {
                activity = kInitializingText;
            }

            // Left sidebar: controls, layers, matcaps and statistics.
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(kSidebarWidth, static_cast<float>(winH)));
            ImGui::Begin("Springls Viewer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::BeginDisabled(quitting);
            ImGui::SetNextItemWidth(-(ImGui::CalcTextSize("Load").x + 2.0f * ImGui::GetStyle().FramePadding.x + ImGui::GetStyle().ItemSpacing.x));
            const bool enter = ImGui::InputText("##mesh", meshPath, sizeof(meshPath), ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::SameLine();
            if (ImGui::Button("Load") || enter) {
                fitPending = true;
                load(meshPath);
            }
            ImGui::BeginDisabled(state.stepFailed);
            if (ImGui::Button(playing ? "Pause" : "Play", ImVec2(80, 0))) {
                playing = !playing;
                worker->setPlaying(playing);
            }
            ImGui::SameLine();
            if (ImGui::Button("Step", ImVec2(80, 0))) worker->requestStep();
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Reset", ImVec2(80, 0))) load(shownMesh.empty() ? requested.mesh : shownMesh);  // the last file that loaded
            if (ImGui::Checkbox("Resampling", &resampling)) worker->setResampling(resampling);
            ImGui::SameLine();
            if (activity) {
                ImGui::TextColored(kWarningColor, "%s", activity);
            } else if (state.stepFailed) {
                ImGui::TextColored(kWarningColor, "Stopped after error");
            } else if (state.busy) {
                ImGui::TextDisabled("Stepping...");
            } else {
                ImGui::TextDisabled("%s", playing ? "Playing" : "Paused");
            }
            // Warnings and errors first, where they are seen without scrolling.
            if (hasMesh) {
                for (const std::string &w : shown->warnings) {
                    ImGui::PushStyleColor(ImGuiCol_Text, kWarningColor);
                    ImGui::TextWrapped("warning: %s", w.c_str());
                    ImGui::PopStyleColor();
                }
            }
            if (!state.error.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, kErrorColor);
                ImGui::TextWrapped("error: %s", state.error.c_str());
                ImGui::PopStyleColor();
            }

            if (ImGui::CollapsingHeader("Grid (applied on Reset)", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SetNextItemWidth(120.0f);
                ImGui::InputFloat("cell scale", &grid.cellScale, 0.1f, 0.5f, "%.2f");
                ImGui::SetNextItemWidth(120.0f);
                ImGui::InputInt("grid (0 = auto)", &grid.gridOverride);
                ImGui::Checkbox("auto depth", &grid.autoDepth);
                ImGui::SameLine();
                ImGui::BeginDisabled(grid.autoDepth);
                ImGui::SetNextItemWidth(-(ImGui::CalcTextSize("max depth").x + ImGui::GetStyle().ItemInnerSpacing.x));
                ImGui::SliderInt("max depth", &grid.maxDepth, 0, AdaptiveGrid3D::kMaxSupportedDepth);
                ImGui::EndDisabled();
                ImGui::RadioButton("Quads", &grid.element, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Triangles", &grid.element, 1);
            }
            ImGui::EndDisabled();

            if (ImGui::CollapsingHeader("Layers", ImGuiTreeNodeFlags_DefaultOpen)) {
                LayerToggle("Iso-surface", &layers.isoSurface, nullptr);
                LayerToggle("Primal springls", &layers.primal, &layers.primalColor);
                LayerToggle("Dual springls", &layers.dual, &layers.dualColor);
                LabeledSlider("springl scale", &layers.springlScale, 0.1f, 1.0f);
                ImGui::Checkbox("Particles", &layers.particles);
                ImGui::SameLine(0.5f * ImGui::GetContentRegionAvail().x);
                ImGui::Checkbox("Input mesh", &layers.inputMesh);
                if (layers.particles) LabeledSlider("particle size", &layers.particleSize, 0.05f, 1.0f);
            }
            if (ImGui::CollapsingHeader("Matcap", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (layers.matcap >= 0 && layers.matcap < static_cast<int>(matcaps.size())) {
                    ImGui::TextUnformatted(matcaps[static_cast<size_t>(layers.matcap)].name.c_str());
                } else {
                    ImGui::TextDisabled("no matcaps found (plain shading)");
                }
                MatcapPicker(renderer, matcaps, layers.matcap);
            }
            if (ImGui::Button("Reset camera")) resetCamera();
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::TextWrapped("LMB orbit | Shift+LMB/MMB pan | RMB/wheel zoom");
            ImGui::PopStyleColor();

            if (ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (hasMesh) {
                    ImGui::TextWrapped("%s", shownMesh.filename().string().c_str());
                    StatsPanel(*shown);
                } else if (!activity) {
                    ImGui::TextDisabled("no mesh loaded");
                }
            }
            ImGui::End();

            // Right sidebar: tri-planar slices of the octree, colored by depth.
            if (hasMesh) {
                ImGui::SetNextWindowPos(ImVec2(std::max(kSidebarWidth, winW - kSliceSidebarWidth), 0));
                ImGui::SetNextWindowSize(ImVec2(kSliceSidebarWidth, static_cast<float>(winH)));
                ImGui::Begin("Grid slices", nullptr,
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);
                const SessionStats &stats = shown->stats;
                springls_gui::TriPlanarSource source;
                source.grid = shown->grid;
                source.isoSurface = &shown->manifold.isoSurface;
                source.extent = stats.baseCells.cast<float>();
                source.defaultSlice = 0.5f * source.extent;
                source.maxDepth = stats.finestDepth;
                source.baseCells = stats.baseCells;
                source.identity = worker->identity();
                source.stamp = shown->stamp;
                slices->draw(source);
                ImGui::End();
            }

            // The view between the sidebars.
            const float viewLeft = kSidebarWidth, viewWidth = std::max(0.0f, winW - kSidebarWidth - (hasMesh ? kSliceSidebarWidth : 0.0f));
            if (!quitting) camera.handleInput();
            if (activity && viewWidth > 0.0f) {
                // On a semi-transparent black panel, so it reads over any surface.
                const ImVec2 size = ImGui::CalcTextSize(activity);
                const ImVec2 at(viewLeft + 0.5f * (viewWidth - size.x), 0.5f * (winH - size.y));
                const ImVec2 pad(12.0f, 8.0f);
                ImDrawList *draw = ImGui::GetBackgroundDrawList();
                draw->AddRectFilled(ImVec2(at.x - pad.x, at.y - pad.y), ImVec2(at.x + size.x + pad.x, at.y + size.y + pad.y), IM_COL32(0, 0, 0, 170), 6.0f);
                draw->AddText(at, IM_COL32(255, 220, 120, 255), activity);
            }

            ImGui::Render();
            if (capturing) {
                target.bind();
            } else {
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }
            glDisable(GL_SCISSOR_TEST);
            glViewport(0, 0, fbW, fbH);
            glClearColor(toybox::palette::kBackground.r, toybox::palette::kBackground.g, toybox::palette::kBackground.b, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            const int viewX = static_cast<int>(std::lround(viewLeft * pixelRatio));
            const int viewW = static_cast<int>(std::lround(viewWidth * pixelRatio));
            if (hasMesh && viewW > 0 && fbH > 0) {  // nothing to draw when minimized or too narrow
                renderer.draw(*shown, layers, camera, viewX, 0, viewW, fbH);
                glDisable(GL_SCISSOR_TEST);
                glViewport(0, 0, fbW, fbH);
            }
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            if (capturing) {
                // Capture once the load and every requested step have been published.
                const bool failed = !state.error.empty() && !state.busy && !state.loadQueued;
                const bool done =
                    hasMesh && !state.busy && !state.loadQueued && state.pendingSteps == 0 && shown->stats.steps >= static_cast<uint64_t>(screenshot.steps);
                if (failed) {
                    status = 1;
                } else if (done && ++settleFrames > kScreenshotSettleFrames) {
                    if (target.writePng(screenshot.path)) {
                        std::cout << "Wrote " << screenshot.path.string() << " (" << fbW << " x " << fbH << ", step " << shown->stats.steps << ")" << std::endl;
                    } else {
                        std::cerr << "error: could not write " << screenshot.path.string() << std::endl;
                        status = 1;
                    }
                    break;
                }
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            } else {
                glfwSwapBuffers(window);
            }
        }
        // Leaving the scope joins the worker (it returns after its running job) before the Session and
        // the GL objects go away.
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return status;
}
}  // namespace viewer
