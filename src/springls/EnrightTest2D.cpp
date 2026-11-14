#include <springls/EnrightTest2D.hpp>

#include <springls/AdaptiveSpringLevelSet2D.hpp>
#include <springls/DoubleweaveLevelSet2D.hpp>
#include <springls/SemiLagrangianDoubleweaveLevelSet2D.hpp>

#include <functional>
#include <map>

namespace springls
{
namespace
{
constexpr double kPiD = 3.14159265358979323846;
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
    sim.setInitialLevelSet(MakeEnrightInitialLevelSet(config));
    sim.setCurvatureWeight(0.0f);
    sim.setVelocityFunction([config](const Vec2 &p) { return EnrightVelocity(config, p); }, 1.0f);
    SpringLevelSet2D::AdvectionSchedule schedule;
    const double period = config.period;
    schedule.weight = [period](double t) { return static_cast<float>(std::cos(kPiD * t / period)); };
    schedule.endTime = period;
    schedule.maxTimeStep = period * config.maxTimeStepFraction;
    sim.setAdvectionSchedule(schedule);
}

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
        case Method::Doubleweave: return "doubleweave";
        case Method::AdaptiveDoubleweave: return "adaptive-doubleweave";
        case Method::SemiLagrangianDoubleweave: return "semi-lagrangian-doubleweave";
    }
    return "";
}

const char *MethodTitle(Method method)
{
    switch (method) {
        case Method::Springls: return "Spring Level Set";
        case Method::AdaptiveSpringls: return "Adaptive Springls";
        case Method::Doubleweave: return "Doubleweave";
        case Method::AdaptiveDoubleweave: return "Adaptive Doubleweave";
        case Method::SemiLagrangianDoubleweave: return "Semi-Lagrangian Doubleweave";
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
    return false;
}

std::unique_ptr<SpringLevelSet2D> MakeSimulation(Method method, int maxDepth)
{
    switch (method) {
        case Method::Springls: return std::make_unique<SpringLevelSet2D>();
        case Method::Doubleweave: return std::make_unique<DoubleweaveLevelSet2D>();
        case Method::AdaptiveSpringls: {
            auto sim = std::make_unique<AdaptiveSpringLevelSet2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::AdaptiveDoubleweave: {
            auto sim = std::make_unique<AdaptiveDoubleweaveLevelSet2D>();
            sim->adaptiveTuning.maxDepth = maxDepth;
            return sim;
        }
        case Method::SemiLagrangianDoubleweave: {
            auto sim = std::make_unique<SemiLagrangianDoubleweaveLevelSet2D>();
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
    const auto doubleweave = [](bool DoubleweaveLevelSet2D::DoubleweaveTuning::*field) {
        return Setter([field](S &s, double v) {
            if (auto *d = dynamic_cast<DoubleweaveLevelSet2D *>(&s)) d->doubleweaveTuning.*field = v != 0.0;
        });
    };
    using A = AdaptiveSpringLevelSet2D::AdaptiveTuning;
    const auto semiLagrangian = [](float SemiLagrangianDoubleweaveLevelSet2D::SemiLagrangianTuning::*field) {
        return Setter([field](S &s, double v) {
            if (auto *d = dynamic_cast<SemiLagrangianDoubleweaveLevelSet2D *>(&s)) d->slTuning.*field = static_cast<float>(v);
        });
    };
    using SL = SemiLagrangianDoubleweaveLevelSet2D::SemiLagrangianTuning;
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
        {"fillFromOther", doubleweave(&DoubleweaveLevelSet2D::DoubleweaveTuning::fillFromOtherConstellation)},
        {"contractCombined", doubleweave(&DoubleweaveLevelSet2D::DoubleweaveTuning::contractAgainstCombinedContour)},
        {"bandCells", adaptive([](A &a, double v) { a.bandCells = static_cast<float>(v); })},
        {"springlCorrection", adaptive([](A &a, double v) { a.springlCorrection = v != 0.0; })},
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
