#include "Check.hpp"

#include <cstring>

// Usage: springls-tests [substring...] runs the tests whose name contains any substring (all without arguments).
int main(int argc, char **argv)
{
    const auto selected = [&](const char *name) {
        if (argc <= 1) return true;
        for (int a = 1; a < argc; a++) {
            if (std::strstr(name, argv[a]) != nullptr) return true;
        }
        return false;
    };
    for (const auto &test : springls_test::Registry()) {
        if (!selected(test.name)) continue;
        const int before = springls_test::FailureCount();
        test.fn();
        std::printf("[%s] %s\n", springls_test::FailureCount() == before ? " ok " : "FAIL", test.name);
    }
    if (springls_test::FailureCount() != 0) {
        std::printf("%d check(s) failed\n", springls_test::FailureCount());
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
