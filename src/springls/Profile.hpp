#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace springls
{
// Opt-in wall-clock profiler: with SPRINGLS_PROFILE set in the environment, every ProfileScope
// adds its duration to a named total, and the totals are printed to stderr at exit (sorted by
// time). Without it, a ProfileScope costs one cached getenv check. Use from the thread that drives
// the simulation (scopes inside OpenMP regions would add up per thread).
class Profile
{
public:
    static bool enabled()
    {
        static const bool on = std::getenv("SPRINGLS_PROFILE") != nullptr;
        return on;
    }
    static void add(const char *name, double seconds)
    {
        Profile &p = instance();
        std::lock_guard<std::mutex> lock(p.mMutex);
        auto &entry = p.mTotals[name];
        entry.first += seconds;
        entry.second++;
    }

private:
    std::mutex mMutex;
    std::map<std::string, std::pair<double, long>> mTotals;
    static Profile &instance()
    {
        static Profile p;
        return p;
    }
    ~Profile()
    {
        if (mTotals.empty()) return;
        std::multimap<double, std::string, std::greater<double>> sorted;
        for (const auto &[name, entry] : mTotals) sorted.emplace(entry.first, name);
        std::fprintf(stderr, "Profile (wall seconds, calls):\n");
        for (const auto &[seconds, name] : sorted) std::fprintf(stderr, "  %-40s %9.2f s %8ld\n", name.c_str(), seconds, mTotals[name].second);
    }
};

class ProfileScope
{
public:
    explicit ProfileScope(const char *name) : mName(Profile::enabled() ? name : nullptr)
    {
        if (mName) mStart = std::chrono::steady_clock::now();
    }
    ~ProfileScope()
    {
        if (mName) Profile::add(mName, std::chrono::duration<double>(std::chrono::steady_clock::now() - mStart).count());
    }
    ProfileScope(const ProfileScope &) = delete;
    ProfileScope &operator=(const ProfileScope &) = delete;

private:
    const char *mName;
    std::chrono::steady_clock::time_point mStart;
};
}  // namespace springls
