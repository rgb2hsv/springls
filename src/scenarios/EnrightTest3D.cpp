#include <scenarios/EnrightTest3D.hpp>

#include <dweave/DweaveLevelSet3D.hpp>
#include <dweave/SemiLagrangianDweaveLevelSet3D.hpp>
#include <levelset/LevelSetMethod3D.hpp>
#include <springl/AdaptiveSpringLevelSet3D.hpp>

#include <functional>
#include <map>

namespace springls
{
namespace
{
constexpr double kPiD = 3.14159265358979323846;
}

VolumeF MakeEnrightInitialLevelSet(const EnrightConfig3D &config)
{
    const int N = config.gridSize;
    const Vec3 center = config.centerVoxels();
    const float radius = config.radiusVoxels();
    VolumeF levelSet(N, N, N);
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) levelSet(i, j, k) = distance(Vec3(i, j, k), center) - radius;
        }
    }
    return levelSet;
}

Vec3 EnrightVelocity(const EnrightConfig3D &config, const Vec3 &position)
{
    const double scale = config.voxelScale();
    const double x = position.x() / scale, y = position.y() / scale, z = position.z() / scale;
    const double sx = std::sin(kPiD * x), sy = std::sin(kPiD * y), sz = std::sin(kPiD * z);
    const double s2x = std::sin(2.0 * kPiD * x), s2y = std::sin(2.0 * kPiD * y), s2z = std::sin(2.0 * kPiD * z);
    const double u = 2.0 * sx * sx * s2y * s2z;
    const double v = -s2x * sy * sy * s2z;
    const double w = -s2x * s2y * sz * sz;
    return Vec3(static_cast<float>(u * scale), static_cast<float>(v * scale), static_cast<float>(w * scale));
}

VolumeV3 MakeEnrightVelocityField(const EnrightConfig3D &config)
{
    const int N = config.gridSize;
    VolumeV3 field(N, N, N);
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) field(i, j, k) = EnrightVelocity(config, Vec3(i, j, k));
        }
    }
    return field;
}

void ConfigureEnright(SpringLevelSet3D &sim, const EnrightConfig3D &config)
{
    sim.setInitialLevelSet(MakeEnrightInitialLevelSet(config));
    sim.setCurvatureWeight(0.0f);
    sim.setVelocityFunction([config](const Vec3 &p) { return EnrightVelocity(config, p); }, 1.0f);
    sim.temporalScheme = SpringLevelSet3D::TemporalScheme::RK4;
    SpringLevelSet3D::AdvectionSchedule schedule;
    const double period = config.period;
    schedule.weight = [period](double t) { return static_cast<float>(std::cos(kPiD * t / period)); };
    schedule.endTime = period;
    schedule.maxTimeStep = period * config.maxTimeStepFraction;
    sim.setAdvectionSchedule(schedule);
}

EnrightMetrics3D MeasureEnright(const SpringLevelSet3D &sim, const EnrightConfig3D &config)
{
    const Manifold3D &m = sim.getManifold();
    const SurfaceMesh3D &iso = m.isoSurface;
    const double scale = config.voxelScale();
    EnrightMetrics3D metrics;
    metrics.time = sim.getSimulationTime();
    metrics.springlCount = m.springlCount();
    if (m.springlConstellations.size() == metrics.springlCount) {
        for (uint8_t c : m.springlConstellations) metrics.constellationSpringlCounts[std::min<size_t>(c, 1)]++;
    } else {
        metrics.constellationSpringlCounts[0] = metrics.springlCount;
    }
    // Divergence theorem over the iso-surface triangles (quads split in two), about the sphere center.
    const Vec3d center = config.centerVoxels().cast<double>();
    const auto &v = iso.vertexes;
    double signedVolume = 0.0;
    const auto addTriangle = [&](uint32_t a, uint32_t b, uint32_t c) {
        const Vec3d p = v[a].cast<double>() - center, q = v[b].cast<double>() - center, r = v[c].cast<double>() - center;
        signedVolume += p.dot(q.cross(r)) / 6.0;
    };
    const int K = iso.elementSize();
    for (size_t f = 0; f < iso.faceCount(); f++) {
        const uint32_t *idx = iso.face(f);
        addTriangle(idx[0], idx[1], idx[2]);
        if (K == 4) addTriangle(idx[2], idx[3], idx[0]);
    }
    metrics.volume = std::abs(signedVolume) / (scale * scale * scale);
    const double exactVolume = 4.0 / 3.0 * kPiD * std::pow(static_cast<double>(config.radius), 3.0);
    metrics.relativeVolumeError = (metrics.volume - exactVolume) / exactVolume;
    const double radius = config.radiusVoxels();
    double shapeErrorSum = 0.0;
    for (const Vec3 &p : v) {
        const double e = std::abs((p.cast<double>() - center).norm() - radius);
        shapeErrorSum += e;
        metrics.maxShapeError = std::max(metrics.maxShapeError, e);
    }
    metrics.meanShapeError = v.empty() ? std::numeric_limits<double>::infinity() : shapeErrorSum / v.size() / scale;
    metrics.maxShapeError /= scale;
    return metrics;
}

bool IsAdaptive(Method3D method)
{
    return method == Method3D::AdaptiveSpringls || method == Method3D::AdaptiveDweave || method == Method3D::SemiLagrangianDweave ||
           method == Method3D::AdaptiveLevelSet;
}
int DefaultGridSize(Method3D method) { return IsAdaptive(method) ? 32 : 128; }

const char *MethodKey(Method3D method)
{
    switch (method) {
        case Method3D::Springls: return "springls";
        case Method3D::Dweave: return "dweave";
        case Method3D::AdaptiveSpringls: return "adaptive-springls";
        case Method3D::AdaptiveDweave: return "adaptive-dweave";
        case Method3D::SemiLagrangianDweave: return "semi-lagrangian-dweave";
        case Method3D::LevelSet: return "levelset";
        case Method3D::AdaptiveLevelSet: return "adaptive-levelset";
    }
    return "";
}

const char *MethodTitle(Method3D method)
{
    switch (method) {
        case Method3D::Springls: return "Spring Level Set 3D";
        case Method3D::Dweave: return "DoubleWeave 3D";
        case Method3D::AdaptiveSpringls: return "Adaptive Springls 3D";
        case Method3D::AdaptiveDweave: return "Adaptive DoubleWeave 3D";
        case Method3D::SemiLagrangianDweave: return "Semi-Lagrangian DoubleWeave 3D";
        case Method3D::LevelSet: return "Level Set 3D";
        case Method3D::AdaptiveLevelSet: return "Adaptive Level Set 3D";
    }
    return "";
}

bool ParseMethod(const std::string &key, Method3D &method)
{
    for (Method3D m : kAllMethods3D) {
        if (key == MethodKey(m)) {
            method = m;
            return true;
        }
    }
    for (Method3D m : kLevelSetMethods3D) {
        if (key == MethodKey(m)) {
            method = m;
            return true;
        }
    }
    return false;
}

const char *ElementTypeKey(ElementType type) { return type == ElementType::Quad ? "quad" : "triangle"; }

bool ParseElementType(const std::string &key, ElementType &type)
{
    if (key == "quad" || key == "quads") {
        type = ElementType::Quad;
    } else if (key == "triangle" || key == "triangles" || key == "tri") {
        type = ElementType::Triangle;
    } else {
        return false;
    }
    return true;
}

std::unique_ptr<SpringLevelSet3D> MakeSimulation(Method3D method, ElementType elementType, int maxDepth)
{
    switch (method) {
        case Method3D::Springls: return std::make_unique<SpringLevelSet3D>(elementType);
        case Method3D::Dweave: return std::make_unique<DweaveLevelSet3D>(elementType);
        case Method3D::AdaptiveSpringls: {
            auto sim = std::make_unique<AdaptiveSpringLevelSet3D>(elementType);
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method3D::AdaptiveDweave: {
            auto sim = std::make_unique<AdaptiveDweaveLevelSet3D>(elementType);
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method3D::SemiLagrangianDweave: {
            auto sim = std::make_unique<SemiLagrangianDweaveLevelSet3D>(elementType);
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method3D::LevelSet: return std::make_unique<LevelSetMethod3D>(elementType);
        case Method3D::AdaptiveLevelSet: {
            auto sim = std::make_unique<AdaptiveLevelSetMethod3D>(elementType);
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
    }
    return nullptr;
}

namespace
{
using Setter = std::function<void(SpringLevelSet3D &, double)>;
const std::map<std::string, Setter> &TuningSetters()
{
    using T = SpringLevelSet3D::Tuning;
    const auto f = [](float T::*field) { return Setter([field](SpringLevelSet3D &s, double v) { s.tuning.*field = static_cast<float>(v); }); };
    const auto i = [](int T::*field) { return Setter([field](SpringLevelSet3D &s, double v) { s.tuning.*field = static_cast<int>(v); }); };
    const auto b = [](bool T::*field) { return Setter([field](SpringLevelSet3D &s, double v) { s.tuning.*field = v != 0.0; }); };
    const auto adaptive = [](auto apply) {
        return Setter([apply](SpringLevelSet3D &s, double v) {
            if (auto *a = dynamic_cast<AdaptiveSpringLevelSet3D *>(&s)) apply(a->adaptiveTuning, v);
        });
    };
    using A = AdaptiveSpringLevelSet3D::AdaptiveTuning;
    const auto semiLagrangian = [](float SemiLagrangianDweaveLevelSet3D::SemiLagrangianTuning::*field) {
        return Setter([field](SpringLevelSet3D &s, double v) {
            if (auto *d = dynamic_cast<SemiLagrangianDweaveLevelSet3D *>(&s)) d->slTuning.*field = static_cast<float>(v);
        });
    };
    using SL = SemiLagrangianDweaveLevelSet3D::SemiLagrangianTuning;
    static const std::map<std::string, Setter> setters = {
        {"armRestLength", f(&T::armRestLength)},
        {"neighborRestSeparation", f(&T::neighborRestSeparation)},
        {"springConstant", f(&T::springConstant)},
        {"sharpness", f(&T::sharpness)},
        {"nearestNeighborDistance", f(&T::nearestNeighborDistance)},
        {"maxNearestNeighbors", i(&T::maxNearestNeighbors)},
        {"relaxIterations", i(&T::relaxIterations)},
        {"relaxTimeStep", f(&T::relaxTimeStep)},
        {"fillDistance", f(&T::fillDistance)},
        {"contractDistance", f(&T::contractDistance)},
        {"maxArmLength", f(&T::maxArmLength)},
        {"minArea", f(&T::minArea)},
        {"maxArea", f(&T::maxArea)},
        {"minAspectRatio", f(&T::minAspectRatio)},
        {"maxFillPasses", i(&T::maxFillPasses)},
        {"fixNormals", b(&T::fixNormals)},
        {"signedEvolveIterations", i(&T::signedEvolveIterations)},
        {"unsignedMaxDistance", f(&T::unsignedMaxDistance)},
        {"maxStepVoxels", f(&T::maxStepVoxels)},
        {"substepsPerStep", i(&T::advectionSubstepsPerStep)},
        {"refineIterations", i(&T::refineIterations)},
        {"refineProximity", f(&T::refineProximity)},
        {"refineStepSize", f(&T::refineStepSize)},
        {"bandCells", adaptive([](A &a, double v) { a.bandCells = static_cast<float>(v); })},
        {"springlCorrection", Setter([](SpringLevelSet3D &s, double v) {
             s.tuning.springlCorrection = v != 0.0;
             if (auto *a = dynamic_cast<AdaptiveSpringLevelSet3D *>(&s)) a->adaptiveTuning.springlCorrection = v != 0.0;
         })},
        {"activeCells", adaptive([](A &a, double v) { a.activeCells = static_cast<float>(v); })},
        {"activeMargin", adaptive([](A &a, double v) { a.activeMargin = static_cast<float>(v); })},
        {"bandWidth", adaptive([](A &a, double v) { a.bandWidth = static_cast<float>(v); })},
        {"capsuleCells", semiLagrangian(&SL::capsuleCells)},
        {"maxStrain", semiLagrangian(&SL::maxStrain)},
        {"maxStepCells", semiLagrangian(&SL::maxStepCells)},
        {"minComponentCells", semiLagrangian(&SL::minComponentCells)},
    };
    return setters;
}
}  // namespace

bool SetTuning(SpringLevelSet3D &sim, const std::string &key, double value)
{
    const auto &setters = TuningSetters();
    const auto it = setters.find(key);
    if (it == setters.end()) return false;
    it->second(sim, value);
    return true;
}

std::vector<std::string> TuningKeys3D()
{
    std::vector<std::string> keys;
    for (const auto &entry : TuningSetters()) keys.push_back(entry.first);
    return keys;
}
}  // namespace springls
