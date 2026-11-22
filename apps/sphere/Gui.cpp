#include "Gui.hpp"
#include "../common/Palette.hpp"

#include "Camera.hpp"
#include "GL.hpp"
#include "Offscreen.hpp"
#include "Renderer.hpp"

#include <core/Common.hpp>
#include <core/SphereSimulation.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sphere_gui
{
using toybox::SphereSimulation;
using toybox::SphereSnapshot;
using toybox::Vec3f;
namespace
{
constexpr float kPanelWidth = 340.0f;
constexpr int kScreenshotWidth = 1600, kScreenshotHeight = 900;
constexpr int kScreenshotSettleFrames = 3;  // ImGui lays out new windows over the first frames
constexpr const char *kDefaultMatcap = "clay_studio";
constexpr float kDefaultArrowSpacing = 2.0f, kDefaultArrowLength = 0.5f;
// Following the object, the view starts framed on it: the half-extent of the view's shorter side is
// kFollowRadii object radii.
constexpr float kFollowRadii = 3.0f;
constexpr float kDefaultIsoSpacing = 0.5f;  // length scales
const ImVec4 kWarningColor(1.0f, 0.8f, 0.3f, 1.0f);
const ImVec4 kErrorColor(1.0f, 0.4f, 0.35f, 1.0f);

void GlfwError(int code, const char *description) { std::cerr << "GLFW error " << code << ": " << description << std::endl; }

// Creates an OpenGL 3.3 core window (hidden when !visible) and makes it current (as the mesh window's).
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
#ifdef SPRINGLS_GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        error = "could not initialize GLEW";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }
    glGetError();  // GLEW may leave GL_INVALID_ENUM behind
#endif
    return window;
}

// Owns the SphereSimulation and runs its init and steps on a thread of its own, taking the snapshot there
// after each (snapshot() is not safe to call while a step runs). The UI posts requests (step, play, arrow
// settings, quit) and picks up the newest snapshot; it never touches the simulation. The destructor joins
// the thread.
class Worker
{
public:
    struct Status {
        bool initialized = false;  // init succeeded
        bool failed = false;       // init or a step failed: error says why; nothing more runs
        bool busy = false;         // init, a step or a snapshot is running
        bool playing = false;
        bool complete = false;  // the loop is done: Play and Step do nothing
        bool exited = false;    // the thread returned (after requestQuit)
        int pendingSteps = 0;
        std::string error;
    };

    Worker(const toybox::Entry &entry, const toybox::RunOptions &options, bool playing, float arrowSpacing, float arrowLength)
        : mEntry(entry), mOptions(options), mPlaying(playing), mArrowSpacing(arrowSpacing), mArrowLength(arrowLength), mThread([this] { run(); })
    {}
    ~Worker()
    {
        requestQuit();
        mThread.join();
    }
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    void requestSteps(int count)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (!mFailed && !mComplete) mPendingSteps += count;
        }
        mChanged.notify_all();
    }
    void setPlaying(bool playing)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mPlaying = playing && !mFailed && !mComplete;
        }
        mChanged.notify_all();
    }
    // Used by the next snapshot; while idle, a new snapshot is taken right away.
    void setArrows(float spacing, float length)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mArrowSpacing = spacing;
            mArrowLength = length;
            mResnapshot = true;
        }
        mChanged.notify_all();
    }
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
        s.initialized = mInitialized;
        s.failed = mFailed;
        s.busy = mBusy;
        s.playing = mPlaying;
        s.complete = mComplete;
        s.exited = mExited;
        s.pendingSteps = mPendingSteps;
        s.error = mError;
        return s;
    }
    // Sets `out` to the newest snapshot if one was published since the last call.
    bool fetch(std::shared_ptr<const SphereSnapshot> &out)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mFetched == mPublished) return false;
        out = mLatest;
        mFetched = mPublished;
        return true;
    }

private:
    const toybox::Entry &mEntry;
    toybox::RunOptions mOptions;
    std::unique_ptr<SphereSimulation> mSimulation;  // only the worker thread touches it
    mutable std::mutex mMutex;
    std::condition_variable mChanged;
    bool mPlaying;
    float mArrowSpacing, mArrowLength;
    bool mResnapshot = false;
    bool mQuit = false, mBusy = true, mInitialized = false, mFailed = false, mComplete = false, mExited = false;
    int mPendingSteps = 0;
    std::string mError;
    std::shared_ptr<const SphereSnapshot> mLatest;
    uint64_t mPublished = 0, mFetched = 0;
    std::thread mThread;  // last, so it starts after every other member is initialized

    // Takes the snapshot with the current arrow settings and publishes it; called unlocked.
    void publish(std::unique_lock<std::mutex> &lock)
    {
        lock.lock();
        const double spacing = mArrowSpacing, length = mArrowLength;
        mResnapshot = false;
        lock.unlock();
        auto snapshot = std::make_shared<const SphereSnapshot>(mSimulation->snapshot(spacing, length));
        lock.lock();
        mLatest = std::move(snapshot);
        mPublished++;
        lock.unlock();
    }

    void run()
    {
        // OpenMP's thread count is per thread: set it here, on the thread that runs init and every step.
        springls::SetThreads(mOptions.sphere.threads);
        std::unique_lock<std::mutex> lock(mMutex, std::defer_lock);
        std::string error;
        try {
            std::unique_ptr<toybox::Simulation> made = mEntry.make(mOptions);
            auto *simulation = dynamic_cast<SphereSimulation *>(made.get());
            if (!simulation) {
                error = "the entry does not make a sphere simulation";
            } else if (simulation->init(error)) {
                made.release();
                mSimulation.reset(simulation);
                publish(lock);
            }
        } catch (const std::exception &e) {
            error = std::string("init failed: ") + e.what();
        } catch (...) {
            error = "init failed: unknown exception";
        }
        lock.lock();
        mBusy = false;
        if (mSimulation && error.empty()) {
            mInitialized = true;
            mComplete = mSimulation->complete();
        } else {
            mFailed = true;
            mError = error.empty() ? std::string("init failed") : error;
        }
        if (mFailed || mComplete) {
            mPlaying = false;
            mPendingSteps = 0;
        }
        while (true) {
            const auto canStep = [this] { return mInitialized && !mFailed && !mComplete && (mPlaying || mPendingSteps > 0); };
            mChanged.wait(lock, [&] { return mQuit || canStep() || (mInitialized && !mFailed && mResnapshot); });
            if (mQuit) break;
            const bool stepping = canStep();
            // A step requested while playing is consumed by the next step too, so none are left over after Pause.
            if (stepping && mPendingSteps > 0) mPendingSteps--;
            mBusy = true;
            lock.unlock();

            error.clear();
            bool complete = false;
            try {
                if (stepping) mSimulation->step();
                complete = mSimulation->complete();
                publish(lock);
            } catch (const std::exception &e) {
                error = std::string("step failed: ") + e.what();
            } catch (...) {
                error = "step failed: unknown exception";
            }

            lock.lock();
            mBusy = false;
            if (!error.empty()) {
                // The simulation may be inconsistent after a step threw: keep the error and stop.
                mError = error;
                mFailed = true;
            }
            mComplete = complete;
            if (mFailed || mComplete) {
                mPlaying = false;
                mPendingSteps = 0;
            }
        }
        mExited = true;
    }
};

const char *ObjectName(springls::LoopObject object) { return object == springls::LoopObject::Star ? "star" : "puzzle piece"; }

// The mesh's bounds and the camera that follows the object or frames the whole mesh.
struct View {
    Camera camera;
    Vec3f lo = Vec3f::Zero(), hi = Vec3f::Zero(), center = Vec3f::Zero();
    bool framed = false;
    bool follow = true;
    float zoom = 0.0f;                      // > 0: the first framing's distance is divided by it (--zoom)
    Vec3f followedNormal = Vec3f::UnitZ();  // the surface normal at the object center the camera last followed

    void setBounds(const SphereSnapshot &snap)
    {
        if (snap.vertices.empty()) return;
        lo = hi = snap.vertices.front();
        for (const Vec3f &p : snap.vertices) {
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
        center = 0.5f * (lo + hi);
    }
    // Looks at the object center from outside, along the surface normal there.
    void lookAtObject(const SphereSnapshot &snap)
    {
        camera.target = snap.objectCenter;
        camera.lookFrom(snap.objectNormal);
        followedNormal = snap.objectNormal;
    }
    // Frames the whole mesh: from the default direction, or (following) from outside the object, far enough
    // back that the mesh stays in view.
    void reset(const SphereSnapshot *snap)
    {
        camera.fit(lo, hi);
        if (follow && snap) {
            camera.distance += (snap->objectCenter - center).norm();
            lookAtObject(*snap);
        }
    }
    // Frames the object: the half-extent of the view's shorter side is kFollowRadii object radii.
    void aim(const SphereSnapshot &snap)
    {
        camera.fit(lo, hi);
        const float halfExtent = kFollowRadii * snap.objectRadius;
        camera.distance = std::max(halfExtent / std::tan(0.5f * camera.fovY), 1e-3f * camera.radius);
        lookAtObject(snap);
    }
    // Keeps the object in view: the target moves with it and the camera turns as the surface normal there
    // does, so the user's orbit offset and zoom are kept.
    void track(const SphereSnapshot &snap)
    {
        if (followedNormal.norm() > 0.0f && snap.objectNormal.norm() > 0.0f) camera.turn(Eigen::Quaternionf::FromTwoVectors(followedNormal, snap.objectNormal));
        camera.target = snap.objectCenter;
        followedNormal = snap.objectNormal;
    }
    // A new snapshot: frame the first one (on the object when following), then follow the object if asked.
    void update(const SphereSnapshot &snap)
    {
        if (!framed) {
            setBounds(snap);
            if (follow) {
                aim(snap);
            } else {
                reset(nullptr);
            }
            if (zoom > 0.0f) camera.distance /= zoom;
            framed = true;
        } else if (follow) {
            track(snap);
        }
    }
    void draw(Renderer &renderer, const Layers &layers, int x, int y, int width, int height) const
    {
        renderer.draw(layers, camera.view(), camera.projection(static_cast<float>(width) / static_cast<float>(std::max(height, 1))), x, y, width, height);
    }
};

Layers InitialLayers(const toybox::Invocation &inv, int matcap)
{
    Layers layers;
    layers.matcap = matcap;
    layers.isoSpacing = kDefaultIsoSpacing;
    for (const std::string &hidden : inv.run.sphere.hidden) {
        if (hidden == "regions") layers.regions = false;
        if (hidden == "isolines") layers.isolines = false;
        if (hidden == "springls") layers.springls = false;
        if (hidden == "contour") layers.contour = false;
        if (hidden == "arrows") layers.arrows = false;
        if (hidden == "reference") layers.reference = false;
        if (hidden == "particles") layers.particles = false;
        if (hidden == "vertices") layers.vertices = false;
        if (hidden == "normals") layers.normals = false;
        if (hidden == "wireframe") layers.wireframe = false;
    }
    return layers;
}

int FindMatcap(const std::vector<viewer::Matcap> &matcaps, const std::string &requested)
{
    const std::string name = requested.empty() ? kDefaultMatcap : requested;
    for (size_t i = 0; i < matcaps.size(); i++)
        if (matcaps[i].name == name) return static_cast<int>(i);
    if (!matcaps.empty()) std::cerr << "warning: no matcap named '" << name << "'" << std::endl;
    return matcaps.empty() ? -1 : 0;
}

// Steps the --steps option asks for (0 for --steps 0); all of them (the loop's end comes first) when not given.
uint64_t TargetSteps(const toybox::Invocation &inv) { return inv.stepsGiven ? inv.limits.maxSteps : std::numeric_limits<uint64_t>::max(); }

// Waits (without a frame loop) until the worker is idle with no steps queued, and returns its status.
Worker::Status WaitIdle(Worker &worker)
{
    while (true) {
        const Worker::Status s = worker.status();
        if (!s.busy && s.pendingSteps == 0 && !s.playing) return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void Panel(const toybox::Invocation &inv, Worker &worker, const Worker::Status &state, const SphereSnapshot *snap, Layers &layers, View &view,
           float &arrowSpacing, float &arrowLength, bool quitting, float x, float height)
{
    ImGui::SetNextWindowPos(ImVec2(x, 0));
    ImGui::SetNextWindowSize(ImVec2(kPanelWidth, height));
    ImGui::Begin("Sphere", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);
    ImGui::Text("%s: %s, %s flow", inv.scenario == toybox::Scenario::Stone ? "Stone" : "Sphere", ObjectName(inv.run.sphere.session.config.object),
                inv.run.sphere.session.config.flow == springls::LoopFlow::Hodge ? "Hodge" : "rotation");
    if (!state.error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kErrorColor);
        ImGui::TextWrapped("error: %s", state.error.c_str());
        ImGui::PopStyleColor();
    }
    if (snap) ImGui::Text("t = %.2f / %.2f", snap->time, snap->period);
    ImGui::BeginDisabled(quitting || !state.initialized || state.failed || state.complete);
    if (ImGui::Button(state.playing ? "Pause" : "Play", ImVec2(80, 0))) worker.setPlaying(!state.playing);
    ImGui::SameLine();
    if (ImGui::Button("Step", ImVec2(80, 0))) worker.requestSteps(1);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (quitting) {
        ImGui::TextColored(kWarningColor, "Finishing the step...");
    } else if (!state.initialized && !state.failed) {
        ImGui::TextColored(kWarningColor, "Initializing...");
    } else if (state.failed) {
        ImGui::TextColored(kWarningColor, "Stopped");
    } else if (state.complete) {
        ImGui::TextDisabled("Complete");
    } else if (state.busy) {
        ImGui::TextDisabled("Stepping...");
    } else {
        ImGui::TextDisabled("%s", state.playing ? "Playing" : "Paused");
    }

    if (snap) {
        const springls::LoopScenario::Metrics &m = snap->metrics;
        ImGui::Separator();
        ImGui::Text("step %llu, springls %zu", static_cast<unsigned long long>(snap->steps), snap->springls.size());
        ImGui::Text("area %+.3f%%", m.relativeAreaErrors.empty() ? 0.0 : 100.0 * m.relativeAreaErrors.front());
        ImGui::Text("contracted %zu, filled %zu", snap->contractedTotal, snap->filledTotal);
        if (!snap->complete) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
        ImGui::Text("shape vs initial (exact at t=T)");
        ImGui::Text("  %.3f / %.3f h (mean / max)", m.meanShapeError, m.maxShapeError);
        if (!snap->complete) ImGui::PopStyleVar();
        ImGui::Text("reference %.3f / %.3f h, gap %.2f h", m.referenceMeanDistance, m.referenceMaxDistance, m.referenceMaxGap);
    }

    ImGui::Separator();
    const float column = 0.5f * ImGui::GetContentRegionAvail().x;
    ImGui::Checkbox("regions", &layers.regions);
    ImGui::SameLine(column);
    ImGui::Checkbox("isolines", &layers.isolines);
    ImGui::Checkbox("springls", &layers.springls);
    ImGui::SameLine(column);
    ImGui::Checkbox("contour", &layers.contour);
    ImGui::Checkbox("arrows", &layers.arrows);
    ImGui::SameLine(column);
    ImGui::Checkbox("reference", &layers.reference);
    ImGui::Checkbox("particles", &layers.particles);
    ImGui::SameLine(column);
    ImGui::Checkbox("vertices", &layers.vertices);
    ImGui::Checkbox("normals", &layers.normals);
    ImGui::SameLine(column);
    ImGui::Checkbox("wireframe", &layers.wireframe);
    const float labelWidth = ImGui::CalcTextSize("arrow spacing").x + ImGui::GetStyle().ItemInnerSpacing.x;
    ImGui::SetNextItemWidth(-labelWidth);
    ImGui::SliderFloat("iso spacing", &layers.isoSpacing, 0.1f, 2.0f, "%.2f h");
    ImGui::SetNextItemWidth(-labelWidth);
    ImGui::SliderFloat("arrow spacing", &arrowSpacing, 0.5f, 8.0f, "%.1f H");
    const bool spacingDone = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SetNextItemWidth(-labelWidth);
    ImGui::SliderFloat("arrow length", &arrowLength, 0.1f, 4.0f, "%.2f H");
    const bool lengthDone = ImGui::IsItemDeactivatedAfterEdit();
    if ((spacingDone || lengthDone) && !quitting) worker.setArrows(arrowSpacing, arrowLength);
    if (ImGui::Checkbox("follow object", &view.follow) && view.follow && snap) view.aim(*snap);
    ImGui::SameLine(column);
    if (ImGui::Button("reset view")) view.reset(snap);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("H = input edge, h = refined edge");
    ImGui::TextWrapped("LMB orbit | Shift+LMB/MMB pan | RMB/wheel zoom");
    ImGui::PopStyleColor();
    ImGui::End();
}

// --frames: the view only, frame_NNNNNN.png every `every` steps and at the last.
int RunFrames(const toybox::Invocation &inv, Worker &worker, Renderer &renderer, Layers layers)
{
    const int size = std::max(16, static_cast<int>(std::lround(inv.frameSize)));
    springls_gui::OffscreenTarget target;
    if (!target.init(size, size)) {
        std::cerr << "error: could not create a " << size << " x " << size << " offscreen framebuffer" << std::endl;
        return 1;
    }
    const uint64_t targetSteps = TargetSteps(inv);
    const uint64_t every = static_cast<uint64_t>(std::max(1, inv.run.sphere.every));
    const double frameTime = inv.run.sphere.frameTime;
    long long lastSlot = -1;  // --frame-time: the multiple of frameTime the last frame was taken in
    View view;
    view.follow = !inv.noFollow;
    view.zoom = inv.zoom;
    bool created = false;  // the directory, once init succeeded (a failed setup leaves none behind)
    std::shared_ptr<const SphereSnapshot> snap;
    size_t written = 0;
    while (true) {
        const Worker::Status state = WaitIdle(worker);
        if (state.failed) {
            std::cerr << "error: " << state.error << std::endl;
            return 1;
        }
        // Every init and step publishes its snapshot before the worker reports idle.
        if (!worker.fetch(snap) || !snap) {
            std::cerr << "error: no snapshot after step " << (snap ? snap->steps : 0) << std::endl;
            return 1;
        }
        if (!created) {
            std::error_code ec;
            std::filesystem::create_directories(inv.frames, ec);
            if (ec) {
                std::cerr << "error: could not create " << inv.frames.string() << ": " << ec.message() << std::endl;
                return 1;
            }
            created = true;
        }
        view.update(*snap);
        const bool last = snap->steps >= targetSteps || state.complete;
        // --frame-time: one image per time slot passed since the last frame (a step longer than a slot, as the
        // flow slows to its turn, repeats its image), so the frames play back evenly in time.
        long long copies = last ? 1 : 0;
        if (frameTime > 0.0) {
            const long long slot = static_cast<long long>(std::floor(snap->time / frameTime));
            if (slot > lastSlot) {
                copies = (lastSlot < 0) ? 1 : slot - lastSlot;
                lastSlot = slot;
            }
        } else if (snap->steps % every == 0) {
            copies = 1;
        }
        if (copies > 0) {
            renderer.setSnapshot(*snap);
            target.bind();
            glDisable(GL_SCISSOR_TEST);
            glViewport(0, 0, size, size);
            glClearColor(toybox::palette::kBackground.r, toybox::palette::kBackground.g, toybox::palette::kBackground.b, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            view.draw(renderer, layers, 0, 0, size, size);
            for (long long c = 0; c < copies; c++) {
                // By step with --every; by frame number with --frame-time (repeats are separate images).
                char name[32];
                std::snprintf(name, sizeof(name), "frame_%06llu.png", static_cast<unsigned long long>(frameTime > 0.0 ? written : snap->steps));
                const std::filesystem::path path = inv.frames / name;
                if (!target.writePng(path)) {
                    std::cerr << "error: could not write " << path.string() << std::endl;
                    return 1;
                }
                written++;
            }
        }
        if (last) {
            std::cout << "Wrote " << written << " frames to " << inv.frames.string() << " (" << size << " x " << size << ", step " << snap->steps << ")"
                      << std::endl;
            return 0;
        }
        worker.requestSteps(1);
    }
}
}  // namespace

int RunSphereGui(const toybox::Invocation &inv)
{
    const bool framing = !inv.frames.empty();
    const bool capturing = framing || !inv.screenshot.empty();
    const uint64_t targetSteps = TargetSteps(inv);
    std::string error;
    int width = kScreenshotWidth, height = kScreenshotHeight;
    if (!capturing) {
        width = 1600;
        height = 1000;
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
    GLFWwindow *window = CreateWindow(width, height, !capturing, "Springls: sphere", error);
    if (!window) {
        std::cerr << "error: " << error << std::endl;
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
        const std::vector<viewer::Matcap> matcaps = viewer::LoadMatcaps(matcapError);
        if (!matcapError.empty()) std::cerr << "warning: " << matcapError << std::endl;
        Layers layers = InitialLayers(inv, FindMatcap(matcaps, inv.matcap));
        float arrowSpacing = kDefaultArrowSpacing, arrowLength = kDefaultArrowLength;

        Renderer renderer;
        springls_gui::OffscreenTarget target;
        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if (!renderer.init(matcaps, error)) {
            std::cerr << "error: " << error << std::endl;
            status = 1;
        } else if (capturing && !framing && !target.init(fbW, fbH)) {
            std::cerr << "error: could not create a " << fbW << " x " << fbH << " offscreen framebuffer" << std::endl;
            status = 1;
        }

        // Interactive: --play plays, --steps N runs N steps and waits. --screenshot plays to the end unless
        // --steps is given. Declared after everything it shares a frame with, so it is joined first.
        const bool playing = capturing ? (!framing && !inv.stepsGiven) : (inv.play && !inv.stepsGiven);
        std::unique_ptr<Worker> worker;
        if (status == 0) worker = std::make_unique<Worker>(*inv.entries.front(), inv.run, playing, arrowSpacing, arrowLength);
        if (status == 0 && !framing && inv.stepsGiven && targetSteps > 0) worker->requestSteps(static_cast<int>(std::min<uint64_t>(targetSteps, 1u << 30)));

        if (status == 0 && framing) {
            status = RunFrames(inv, *worker, renderer, layers);
        } else if (status == 0) {
            View view;
            view.follow = !inv.noFollow;
            view.zoom = inv.zoom;
            std::shared_ptr<const SphereSnapshot> shown;
            bool quitting = false, errorPrinted = false;
            int settleFrames = 0;
            while (true) {
                if (capturing) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                glfwPollEvents();
                if (glfwWindowShouldClose(window) && !quitting) {
                    // The running init or step cannot be interrupted: keep drawing until the worker returns.
                    quitting = true;
                    worker->requestQuit();
                }
                const Worker::Status state = worker->status();
                if (quitting && state.exited) break;
                if (worker->fetch(shown) && shown) {
                    renderer.setSnapshot(*shown);
                    view.update(*shown);
                }
                if (state.failed && !errorPrinted) {
                    // Interactive: the error stays in the panel until the window is closed; either way the exit code is 1.
                    std::cerr << "error: " << state.error << std::endl;
                    errorPrinted = true;
                    status = 1;
                    if (capturing) break;
                }

                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                int winW, winH;
                glfwGetWindowSize(window, &winW, &winH);
                glfwGetFramebufferSize(window, &fbW, &fbH);
                const float pixelRatio = (winW > 0) ? static_cast<float>(fbW) / winW : 1.0f;
                Panel(inv, *worker, state, shown.get(), layers, view, arrowSpacing, arrowLength, quitting, 0.0f, static_cast<float>(winH));
                if (!quitting) view.camera.handleInput();
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
                // The panel on the left, the view to its right.
                const int viewX = static_cast<int>(std::lround(kPanelWidth * pixelRatio));
                const int viewW = std::max(0, fbW - viewX);
                if (renderer.hasSnapshot() && viewW > 0 && fbH > 0) {
                    view.draw(renderer, layers, viewX, 0, viewW, fbH);
                    glViewport(0, 0, fbW, fbH);
                }
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                if (capturing) {
                    // Capture once every requested step has been published (or the loop ended).
                    const bool done = shown && state.initialized && !state.busy && state.pendingSteps == 0 && !state.playing &&
                                      (shown->steps >= targetSteps || state.complete);
                    if (done && ++settleFrames > kScreenshotSettleFrames) {
                        if (target.writePng(inv.screenshot)) {
                            std::cout << "Wrote " << inv.screenshot.string() << " (" << fbW << " x " << fbH << ", step " << shown->steps << ")" << std::endl;
                        } else {
                            std::cerr << "error: could not write " << inv.screenshot.string() << std::endl;
                            status = 1;
                        }
                        break;
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                } else {
                    glfwSwapBuffers(window);
                }
            }
        }
        // Leaving the scope joins the worker before the simulation and the GL objects go away.
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return status;
}
}  // namespace sphere_gui
