#pragma once
#include <cmath>
#include <cstdio>
#include <vector>

namespace springls_test
{
struct TestCase {
    const char *name;
    void (*fn)();
};

inline std::vector<TestCase> &Registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

inline int &FailureCount()
{
    static int failures = 0;
    return failures;
}

struct Registrar {
    Registrar(const char *name, void (*fn)()) { Registry().push_back({name, fn}); }
};
}  // namespace springls_test

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++springls_test::FailureCount();                                     \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                                                           \
    do {                                                                                                                                                \
        const double checkA = static_cast<double>(a), checkB = static_cast<double>(b);                                                                  \
        if (!(std::fabs(checkA - checkB) <= static_cast<double>(tol))) {                                                                                \
            std::printf("%s:%d: CHECK_NEAR failed: %s = %g, %s = %g (tol %g)\n", __FILE__, __LINE__, #a, checkA, #b, checkB, static_cast<double>(tol)); \
            ++springls_test::FailureCount();                                                                                                            \
        }                                                                                                                                               \
    } while (0)

// Defines and registers a test function: TEST(name) { CHECK(...); }
#define TEST(name)                                                 \
    static void name();                                            \
    static springls_test::Registrar name##Registrar(#name, &name); \
    static void name()
