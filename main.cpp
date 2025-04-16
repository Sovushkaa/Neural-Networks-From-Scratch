#include "Except.h"
#include "tests.h"

using namespace CNetworks;

int main() {
    try {
        RunALLTests();
    } catch (...) {
        except::react();
    }
    return 0;
}
