#include <scenarios/EnrightTest2D.hpp>

#include <dweave/AdaptiveMultiDweaveLevelSet2D.hpp>
#include <dweave/DweaveLevelSet2D.hpp>
#include <dweave/MultiDweaveLevelSet2D.hpp>
#include <dweave/SemiLagrangianDweaveLevelSet2D.hpp>
#include <levelset/AdaptiveMultiActiveContour2D.hpp>
#include <levelset/LevelSetMethod2D.hpp>
#include <levelset/MultiActiveContour2D.hpp>
#include <springl/AdaptiveMultiSpringLevelSet2D.hpp>
#include <springl/AdaptiveSpringLevelSet2D.hpp>
#include <springl/MultiSpringLevelSet2D.hpp>

#include <functional>
#include <map>
#include <numbers>

namespace springls
{
namespace
{
constexpr double kPiD = std::numbers::pi;
}

ImageF MakeEnrightInitialLevelSet(const EnrightConfig2D &config)
{
    const int N = config.gridSize;
    const Vec2 center = config.centerPixels();
    const float radius = config.radiusPixels();
    ImageF levelSet(N, N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) levelSet(i, j) = distance(Vec2(i, j), center) - radius;
    }
    return levelSet;
}

ImageI MakeEnrightInitialLabels(const EnrightConfig2D &config)
{
    const int N = config.gridSize;
    const Vec2 center = config.centerPixels();
    const float radius = config.radiusPixels();
    ImageI labels(N, N, SpringLevelSet2D::kBackgroundLabel);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            const Vec2 d = Vec2(i, j) - center;
            if (d.norm() >= radius) continue;
            labels(i, j) = (config.regions == 1) ? 1 : (d.y() >= 0.0f) ? ((d.x() >= 0.0f) ? 1 : 2) : ((d.x() < 0.0f) ? 3 : 4);
        }
    }
    return labels;
}

float EnrightPartitionDistance(const EnrightConfig2D &config, const Vec2 &position)
{
    const Vec2 d = position - config.centerPixels();
    const float radius = config.radiusPixels();
    const float toCircle = std::abs(d.norm() - radius);
    if (config.regions == 1) return toCircle;
    // Nearest points on the diameters x = 0 and y = 0, clipped to the disk.
    const float toVertical = (d - Vec2(0.0f, std::clamp(d.y(), -radius, radius))).norm();
    const float toHorizontal = (d - Vec2(std::clamp(d.x(), -radius, radius), 0.0f)).norm();
    return std::min({toCircle, toVertical, toHorizontal});
}

ImageF MakeEnrightInitialLabelDistance(const EnrightConfig2D &config)
{
    const int N = config.gridSize;
    ImageF distance(N, N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) distance(i, j) = EnrightPartitionDistance(config, Vec2(i, j));
    }
    return distance;
}

Vec2 EnrightVelocity(const EnrightConfig2D &config, const Vec2 &position)
{
    const double scale = config.pixelScale();
    const double x = position.x() / scale, y = position.y() / scale;
    const double sx = std::sin(kPiD * x), sy = std::sin(kPiD * y);
    const double u = sx * sx * std::sin(2.0 * kPiD * y);
    const double v = -sy * sy * std::sin(2.0 * kPiD * x);
    return Vec2(static_cast<float>(u * scale), static_cast<float>(v * scale));
}

ImageV2 MakeEnrightVelocityField(const EnrightConfig2D &config)
{
    const int N = config.gridSize;
    ImageV2 field(N, N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) field(i, j) = EnrightVelocity(config, Vec2(i, j));
    }
    return field;
}

void ConfigureEnright(SpringLevelSet2D &sim, const EnrightConfig2D &config)
{
    if (sim.isLabeled()) {
        sim.setInitialLabels(MakeEnrightInitialLabels(config), MakeEnrightInitialLabelDistance(config));
        sim.backgroundCurvatureWeight = 0.0f;
    } else {
        sim.setInitialLevelSet(MakeEnrightInitialLevelSet(config));
    }
    sim.setCurvatureWeight(0.0f);
    sim.setVelocityFunction([config](const Vec2 &p) { return EnrightVelocity(config, p); }, 1.0f);
    SpringLevelSet2D::AdvectionSchedule schedule;
    const double period = config.period;
    schedule.weight = [period](double t) { return static_cast<float>(std::cos(kPiD * t / period)); };
    schedule.endTime = period;
    if (config.maxTimeStepFraction > 0.0) schedule.maxTimeStep = period * config.maxTimeStepFraction;
    sim.setAdvectionSchedule(schedule);
}

namespace
{
// Per-region areas and the partition shape error, for the multi-object variant.
EnrightMetrics2D MeasureMultiObject(const SpringLevelSet2D &sim, const EnrightConfig2D &config, EnrightMetrics2D metrics)
{
    const Manifold2D &m = sim.getManifold();
    const double scale = config.pixelScale();
    const int regions = (config.regions == 1) ? 1 : kEnrightRegionCount;
    std::vector<double> signedAreas(regions, 0.0);
    double shapeErrorSum = 0.0;
    size_t vertexCount = 0;
    for (size_t c = 0; c < m.vertexIndexes.size(); c++) {
        Curve curve = m.vertexIndexes[c];
        const int label = m.curveLabels[c];
        if (curve.size() > 1 && curve.front() == curve.back()) curve.pop_back();
        const size_t N = curve.size();
        double signedArea = 0.0;
        for (size_t k = 0; k < N; k++) {
            const Vec2 a = m.vertexLocations[curve[k]], b = m.vertexLocations[curve[(k + 1) % N]];
            signedArea += static_cast<double>(a.x()) * b.y() - static_cast<double>(b.x()) * a.y();
            const double shapeError = EnrightPartitionDistance(config, a);
            shapeErrorSum += shapeError;
            metrics.maxShapeError = std::max(metrics.maxShapeError, shapeError);
            vertexCount++;
        }
        if (label >= 1 && label <= regions) signedAreas[label - 1] += signedArea;
    }
    const double exactArea = kPiD * static_cast<double>(config.radius) * config.radius / regions;
    metrics.area = 0.0;
    for (double signedArea : signedAreas) {
        const double area = 0.5 * std::abs(signedArea) / (scale * scale);
        metrics.labelAreas.push_back(area);
        metrics.labelRelativeAreaErrors.push_back((area - exactArea) / exactArea);
        metrics.area += area;
    }
    metrics.relativeAreaError = (metrics.area - regions * exactArea) / (regions * exactArea);
    metrics.meanShapeError = (vertexCount > 0) ? shapeErrorSum / vertexCount / scale : std::numeric_limits<double>::infinity();
    metrics.maxShapeError /= scale;
    return metrics;
}
}  // namespace

EnrightMetrics2D MeasureEnright(const SpringLevelSet2D &sim, const EnrightConfig2D &config)
{
    const Manifold2D &m = sim.getManifold();
    const double scale = config.pixelScale();
    EnrightMetrics2D metrics;
    metrics.time = sim.getSimulationTime();
    metrics.springlCount = m.springlCount();
    if (m.springlConstellations.size() == metrics.springlCount) {
        for (uint8_t c : m.springlConstellations) metrics.constellationSpringlCounts[c]++;
    } else {
        metrics.constellationSpringlCounts[0] = metrics.springlCount;
    }
    if (sim.isLabeled()) return MeasureMultiObject(sim, config, metrics);
    // Shoelace area over all curves; open curves (touching the domain edge) are closed implicitly.
    double signedArea = 0.0, shapeErrorSum = 0.0;
    size_t vertexCount = 0;
    const Vec2 center = config.centerPixels();
    const double radius = config.radiusPixels();
    for (Curve curve : m.vertexIndexes) {
        if (curve.size() > 1 && curve.front() == curve.back()) curve.pop_back();
        const size_t N = curve.size();
        for (size_t k = 0; k < N; k++) {
            const Vec2 a = m.vertexLocations[curve[k]], b = m.vertexLocations[curve[(k + 1) % N]];
            signedArea += static_cast<double>(a.x()) * b.y() - static_cast<double>(b.x()) * a.y();
            const double shapeError = std::abs(distance(a, center) - radius);
            shapeErrorSum += shapeError;
            metrics.maxShapeError = std::max(metrics.maxShapeError, shapeError);
            vertexCount++;
        }
    }
    metrics.area = 0.5 * std::abs(signedArea) / (scale * scale);
    const double exactArea = kPiD * static_cast<double>(config.radius) * config.radius;
    metrics.relativeAreaError = (metrics.area - exactArea) / exactArea;
    metrics.meanShapeError = (vertexCount > 0) ? shapeErrorSum / vertexCount / scale : std::numeric_limits<double>::infinity();
    metrics.maxShapeError /= scale;
    return metrics;
}

const char *MethodKey(Method method)
{
    switch (method) {
        case Method::Springls: return "springls";
        case Method::AdaptiveSpringls: return "adaptive-springls";
        case Method::Dweave: return "dweave";
        case Method::AdaptiveDweave: return "adaptive-dweave";
        case Method::SemiLagrangianDweave: return "semi-lagrangian-dweave";
        case Method::Mogac: return "multi-levelset";
        case Method::MultiSpringls: return "multi-springls";
        case Method::AdaptiveMultiSpringls: return "adaptive-multi-springls";
        case Method::AdaptiveMogac: return "adaptive-multi-levelset";
        case Method::MultiDweave: return "multi-dweave";
        case Method::AdaptiveMultiDweave: return "adaptive-multi-dweave";
        case Method::LevelSet: return "levelset";
        case Method::AdaptiveLevelSet: return "adaptive-levelset";
    }
    return "";
}

const char *MethodTitle(Method method)
{
    switch (method) {
        case Method::Springls: return "Spring Level Set";
        case Method::AdaptiveSpringls: return "Adaptive Springls";
        case Method::Dweave: return "DoubleWeave";
        case Method::AdaptiveDweave: return "Adaptive DoubleWeave";
        case Method::SemiLagrangianDweave: return "Semi-Lagrangian DoubleWeave";
        case Method::Mogac: return "MOGAC";
        case Method::MultiSpringls: return "Multi-Object Springls";
        case Method::AdaptiveMultiSpringls: return "Adaptive Multi-Object Springls";
        case Method::AdaptiveMogac: return "Adaptive MOGAC";
        case Method::MultiDweave: return "Multi-Object DoubleWeave";
        case Method::AdaptiveMultiDweave: return "Adaptive Multi-Object DoubleWeave";
        case Method::LevelSet: return "Level Set";
        case Method::AdaptiveLevelSet: return "Adaptive Level Set";
    }
    return "";
}

bool ParseMethod(const std::string &key, Method &method)
{
    for (Method m : kAllMethods) {
        if (key == MethodKey(m)) {
            method = m;
            return true;
        }
    }
    for (Method m : kMultiMethods) {
        if (key == MethodKey(m)) {
            method = m;
            return true;
        }
    }
    for (Method m : kLevelSetMethods) {
        if (key == MethodKey(m)) {
            method = m;
            return true;
        }
    }
    return false;
}

bool IsAdaptive(Method method)
{
    switch (method) {
        case Method::AdaptiveSpringls:
        case Method::AdaptiveDweave:
        case Method::SemiLagrangianDweave:
        case Method::AdaptiveMultiSpringls:
        case Method::AdaptiveMogac:
        case Method::AdaptiveMultiDweave:
        case Method::AdaptiveLevelSet: return true;
        case Method::Springls:
        case Method::Dweave:
        case Method::Mogac:
        case Method::MultiSpringls:
        case Method::MultiDweave:
        case Method::LevelSet: return false;
    }
    return false;
}

int DefaultGridSize(Method method) { return IsAdaptive(method) ? kDefaultAdaptiveGridSize : kDefaultUniformGridSize; }

std::unique_ptr<SpringLevelSet2D> MakeSimulation(Method method, int maxDepth)
{
    switch (method) {
        case Method::Springls: return std::make_unique<SpringLevelSet2D>();
        case Method::Dweave: return std::make_unique<DweaveLevelSet2D>();
        case Method::AdaptiveSpringls: {
            auto sim = std::make_unique<AdaptiveSpringLevelSet2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::AdaptiveDweave: {
            auto sim = std::make_unique<AdaptiveDweaveLevelSet2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::SemiLagrangianDweave: {
            auto sim = std::make_unique<SemiLagrangianDweaveLevelSet2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::Mogac: return std::make_unique<MultiActiveContour2D>();
        case Method::MultiSpringls: return std::make_unique<MultiSpringLevelSet2D>();
        case Method::AdaptiveMultiSpringls: {
            auto sim = std::make_unique<AdaptiveMultiSpringLevelSet2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::AdaptiveMogac: {
            auto sim = std::make_unique<AdaptiveMultiActiveContour2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::MultiDweave: return std::make_unique<MultiDweaveLevelSet2D>();
        case Method::AdaptiveMultiDweave: {
            auto sim = std::make_unique<AdaptiveMultiDweaveLevelSet2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::LevelSet: return std::make_unique<LevelSetMethod2D>();
        case Method::AdaptiveLevelSet: {
            auto sim = std::make_unique<AdaptiveLevelSetMethod2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
    }
    return nullptr;
}

namespace
{
using Setter = std::function<void(SpringLevelSet2D &, double)>;
const std::map<std::string, Setter> &TuningSetters()
{
    using S = SpringLevelSet2D;
    const auto f = [](float S::Tuning::*field) { return Setter([field](S &s, double v) { s.tuning.*field = static_cast<float>(v); }); };
    const auto i = [](int S::Tuning::*field) { return Setter([field](S &s, double v) { s.tuning.*field = static_cast<int>(v); }); };
    const auto adaptive = [](auto apply) {
        return Setter([apply](S &s, double v) {
            if (auto *a = dynamic_cast<AdaptiveSpringLevelSet2D *>(&s)) apply(a->adaptiveTuning, v);
        });
    };
    const auto dweave = [](bool DweaveLevelSet2D::DweaveTuning::*field) {
        return Setter([field](S &s, double v) {
            if (auto *d = dynamic_cast<DweaveLevelSet2D *>(&s)) d->dweaveTuning.*field = v != 0.0;
        });
    };
    using A = AdaptiveSpringLevelSet2D::AdaptiveTuning;
    const auto semiLagrangian = [](float SemiLagrangianDweaveLevelSet2D::SemiLagrangianTuning::*field) {
        return Setter([field](S &s, double v) {
            if (auto *d = dynamic_cast<SemiLagrangianDweaveLevelSet2D *>(&s)) d->slTuning.*field = static_cast<float>(v);
        });
    };
    using SL = SemiLagrangianDweaveLevelSet2D::SemiLagrangianTuning;
    static const std::map<std::string, Setter> setters = {
        {"armRestLength", f(&S::Tuning::armRestLength)},
        {"neighborRestSeparation", f(&S::Tuning::neighborRestSeparation)},
        {"springConstant", f(&S::Tuning::springConstant)},
        {"sharpness", f(&S::Tuning::sharpness)},
        {"nearestNeighborDistance", f(&S::Tuning::nearestNeighborDistance)},
        {"relaxIterations", i(&S::Tuning::relaxIterations)},
        {"relaxTimeStep", f(&S::Tuning::relaxTimeStep)},
        {"fillDistance", f(&S::Tuning::fillDistance)},
        {"contractDistance", f(&S::Tuning::contractDistance)},
        {"maxArmLength", f(&S::Tuning::maxArmLength)},
        {"signedEvolveIterations", i(&S::Tuning::signedEvolveIterations)},
        {"unsignedMaxDistance", f(&S::Tuning::unsignedMaxDistance)},
        {"maxStepPixels", f(&S::Tuning::maxStepPixels)},
        {"substepsPerStep", i(&S::Tuning::advectionSubstepsPerStep)},
        {"refineIterations", i(&S::Tuning::refineIterations)},
        {"refineProximity", f(&S::Tuning::refineProximity)},
        {"refineStepSize", f(&S::Tuning::refineStepSize)},
        {"backgroundCurvatureWeight", Setter([](S &s, double v) { s.backgroundCurvatureWeight = static_cast<float>(v); })},
        {"fillFromOther", dweave(&DweaveLevelSet2D::DweaveTuning::fillFromOtherConstellation)},
        {"contractCombined", dweave(&DweaveLevelSet2D::DweaveTuning::contractAgainstCombinedContour)},
        {"bandCells", adaptive([](A &a, double v) { a.bandCells = static_cast<float>(v); })},
        {"maxStepFinestCells", adaptive([](A &a, double v) { a.maxStepCells = static_cast<float>(v); })},
        {"springlCorrection", Setter([](S &s, double v) {
             if (auto *a = dynamic_cast<AdaptiveSpringLevelSet2D *>(&s)) a->adaptiveTuning.springlCorrection = v != 0.0;
             s.tuning.springlCorrection = v != 0.0;
         })},
        {"bandWidth", adaptive([](A &a, double v) { a.bandWidth = static_cast<float>(v); })},
        {"capsuleCells", semiLagrangian(&SL::capsuleCells)},
        {"minBandPixels", semiLagrangian(&SL::minBandPixels)},
        {"maxStrain", semiLagrangian(&SL::maxStrain)},
        {"maxStepCells", semiLagrangian(&SL::maxStepCells)},
        {"minLoopCells", semiLagrangian(&SL::minLoopCells)},
    };
    return setters;
}
}  // namespace

bool SetTuning(SpringLevelSet2D &sim, const std::string &key, double value)
{
    const auto &setters = TuningSetters();
    const auto it = setters.find(key);
    if (it == setters.end()) return false;
    it->second(sim, value);
    return true;
}

std::vector<std::string> TuningKeys()
{
    std::vector<std::string> keys;
    for (const auto &entry : TuningSetters()) keys.push_back(entry.first);
    return keys;
}
}  // namespace springls
