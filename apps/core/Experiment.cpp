#include "Experiment.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace toybox
{
namespace
{
double SecondsSince(std::chrono::steady_clock::time_point start) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
}  // namespace

const char *PacingKey(Pacing pacing)
{
    switch (pacing) {
        case Pacing::Lockstep: return "lockstep";
        case Pacing::EqualCompute: return "equal-compute";
        case Pacing::SameTime: return "same-time";
    }
    return "";
}

const char *PacingTitle(Pacing pacing)
{
    switch (pacing) {
        case Pacing::Lockstep: return "lockstep (one step each)";
        case Pacing::EqualCompute: return "equal compute (least wall-clock time steps next)";
        case Pacing::SameTime: return "same simulation time (furthest behind steps next)";
    }
    return "";
}

bool ParsePacing(const std::string &key, Pacing &pacing)
{
    for (Pacing p : {Pacing::Lockstep, Pacing::EqualCompute, Pacing::SameTime}) {
        if (key == PacingKey(p)) {
            pacing = p;
            return true;
        }
    }
    return false;
}

bool Experiment::init(std::vector<const Entry *> entries, const RunOptions &options, Pacing pacing, const Limits &limits, std::string &error)
{
    mEntries = std::move(entries);
    mOptions = options;
    mPacing = pacing;
    mLimits = limits;
    return restart(error);
}

bool Experiment::restart(std::string &error)
{
    mRuns.clear();
    mStepCount = 0;
    mGeneration++;
    if (mEntries.empty()) {
        error = "no methods selected";
        return false;
    }
    if (!mLimits.resumeFile.empty() && mEntries.size() != 1) {
        error = "resuming from a state needs exactly one method";
        return false;
    }
    std::sort(mLimits.checkpointTimes.begin(), mLimits.checkpointTimes.end());
    for (const Entry *entry : mEntries) {
        Run run;
        run.sim = entry->make(mOptions);
        const auto start = std::chrono::steady_clock::now();
        if (!run.sim->init(error)) return false;
        run.seconds = SecondsSince(start);
        if (!mLimits.resumeFile.empty()) {
            if (!run.sim->supportsCheckpoints()) {
                error = "--resume is not supported for " + entry->title;
                return false;
            }
            if (!run.sim->resume(mLimits.resumeFile, error)) return false;
        }
        run.startSpringls = run.sim->springlCount();
        run.done = run.sim->complete() || run.sim->time() >= mLimits.stopAt || mLimits.maxSteps == 0;
        // Checkpoint times already passed (when resuming) are not written again.
        while (run.nextCheckpoint < mLimits.checkpointTimes.size() && mLimits.checkpointTimes[run.nextCheckpoint] < run.sim->time()) run.nextCheckpoint++;
        mRuns.push_back(std::move(run));
    }
    writeCsvRows();
    return true;
}

bool Experiment::step()
{
    // The one run to advance under EqualCompute / SameTime (ties go to the earlier run).
    const Run *next = nullptr;
    if (mPacing != Pacing::Lockstep) {
        for (const Run &run : mRuns) {
            if (run.done) continue;
            const auto key = [&](const Run &r) { return mPacing == Pacing::EqualCompute ? r.seconds : r.sim->time(); };
            if (!next || key(run) < key(*next)) next = &run;
        }
    }
    bool advanced = false;
    for (Run &run : mRuns) {
        run.stepped = !run.done && (mPacing == Pacing::Lockstep || &run == next);
        if (!run.stepped) continue;
        const auto start = std::chrono::steady_clock::now();
        run.done = !run.sim->step();
        run.stepSeconds = SecondsSince(start);
        run.seconds += run.stepSeconds;
        run.stepsTaken++;
        if (run.sim->time() >= mLimits.stopAt || run.stepsTaken >= mLimits.maxSteps) run.done = true;
        const size_t springls = run.sim->springlCount();
        if (static_cast<double>(springls) > mLimits.maxSpringlGrowth * static_cast<double>(run.startSpringls)) {
            run.done = run.aborted = true;
            std::cout << "Aborted " << run.sim->entry().title << " at t = " << run.sim->time() << " (step " << run.sim->iteration() << "): springls grew from "
                      << run.startSpringls << " to " << springls << std::endl;
        }
        std::string error;
        if (!saveCheckpoints(run, error)) std::cerr << "error: " << error << std::endl;
        advanced = true;
    }
    if (advanced) {
        mStepCount++;
        writeCsvRows();
    }
    return advanced;
}

bool Experiment::saveCheckpoints(Run &run, std::string &error)
{
    if (!run.sim->supportsCheckpoints()) return true;
    while (run.nextCheckpoint < mLimits.checkpointTimes.size() && run.sim->time() >= mLimits.checkpointTimes[run.nextCheckpoint]) {
        const double t = mLimits.checkpointTimes[run.nextCheckpoint++];
        if (!run.sim->saveCheckpoint(mLimits.checkpointDir, t, error)) return false;
    }
    return true;
}

bool Experiment::done() const
{
    for (const Run &run : mRuns) {
        if (!run.done) return false;
    }
    return true;
}

bool Experiment::openCsv(const std::filesystem::path &file)
{
    if (mRuns.empty() || !mRuns.front().sim->supportsCsv()) return false;
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());
    mCsv.open(file, std::ios::binary);  // LF line endings on every platform
    if (!mCsv) return false;
    mRuns.front().sim->csvHeader(mCsv);
    // Multi-object runs also report each region, in a sibling file (the main CSV keeps its columns).
    const Run *labeled = nullptr;
    for (const Run &run : mRuns) {
        if (!labeled && run.sim->hasLabelsCsv()) labeled = &run;
    }
    if (labeled) {
        std::filesystem::path labelsFile = file;
        labelsFile.replace_filename(file.stem().string() + "-labels" + file.extension().string());
        mLabelsCsv.open(labelsFile, std::ios::binary);
        if (!mLabelsCsv) return false;
        labeled->sim->labelsCsvHeader(mLabelsCsv);
    }
    writeCsvRows();
    return true;
}

void Experiment::closeCsv()
{
    if (mCsv.is_open()) mCsv.close();
    if (mLabelsCsv.is_open()) mLabelsCsv.close();
}

void Experiment::writeCsvRows()
{
    if (!mCsv.is_open()) return;
    for (const Run &run : mRuns) {
        if (!run.stepped) continue;  // finished earlier; its last row is already written
        run.sim->csvRow(mCsv, run.stepSeconds);
        if (mLabelsCsv.is_open() && run.sim->hasLabelsCsv()) run.sim->labelsCsvRows(mLabelsCsv);
    }
    mCsv.flush();
    if (mLabelsCsv.is_open()) mLabelsCsv.flush();
}

void Experiment::printSummary(std::ostream &out) const
{
    if (mRuns.empty()) return;
    // Flags are restored, the precision is left at 5 (as the old programs did).
    const auto flags = out.flags();
    mRuns.front().sim->summaryHeader(out, period());
    out << std::setprecision(5);
    for (const Run &run : mRuns) run.sim->summary(out, run.seconds);
    out.flags(flags);
}
}  // namespace toybox
