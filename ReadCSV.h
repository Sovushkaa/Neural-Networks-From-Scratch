#pragma once
#include "linalg.h"

namespace CNetworks {
    std::pair<Matrix, Matrix> ReadCSV(const std::string &filename);
} // namespace CNetworks
