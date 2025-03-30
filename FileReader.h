#pragma once
#include "Network.h"
#include <filesystem>
#include <fstream>

namespace CNetworks {
    class FileReader {
    public:
        FileReader(const std::filesystem::path &filename);
        template <class Data>
        Data read();

    private:
        std::ifstream stream_;
    };

    template <>
    Network FileReader::read<Network>();
    template <>
    Layer FileReader::read<Layer>();
    template <>
    Matrix FileReader::read<Matrix>();
    template <>
    Vector FileReader::read<Vector>();
} // namespace CNetworks
