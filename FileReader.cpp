#include "FileReader.h"
#include <cassert>

namespace CNetworks {
    FileReader::FileReader(const std::filesystem::path &filename)
        : stream_(filename) {
        assert(stream_);
    }

    template <>
    Network FileReader::read<Network>() {
        std::vector<Layer> layers;
        Index layers_number;
        stream_ >> layers_number;
        layers.reserve(layers_number);
        for (Index i = 0; i < layers_number; ++i)
            layers.push_back(read<Layer>());
        return Network(std::move(layers));
    }

    template <>
    Layer FileReader::read<Layer>() {
        Matrix A = read<Matrix>();
        Vector b = read<Vector>();
        return Layer(std::move(A), std::move(b));
    }

    template <>
    Matrix FileReader::read<Matrix>() {
        Index rows, cols;
        stream_ >> rows >> cols;
        Matrix ans;
        ans.resize(rows, cols);
        for (Index i = 0; i < rows; ++i) {
            for (Index j = 0; j < cols; ++j) {
                stream_ >> ans(i, j);
            }
        }
        return ans;
    }

    template <>
    Vector FileReader::read<Vector>() {
        Index rows;
        stream_ >> rows;
        Vector ans;
        ans.resize(rows);
        for (Index i = 0; i < rows; ++i) {
            stream_ >> ans(i);
        }
        return ans;
    }
} // namespace CNetworks
