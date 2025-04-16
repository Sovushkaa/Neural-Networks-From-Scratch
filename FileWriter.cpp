#include "FileWriter.h"
#include <cassert>
#include <iostream>

namespace CNetworks {
    FileWriter::FileWriter(const std::filesystem::path &filename)
        : stream_(filename) {
        assert(stream_);
    }

    template <>
    void FileWriter::write<Network>(const Network &network) {
        stream_ << network.layers_.size() << std::endl;
        ;
        for (const auto &layer : network.layers_) {
            write(layer);
        }
    }

    template <>
    void FileWriter::write<Layer>(const Layer &layer) {
        write(layer.Get_A());
        write(layer.Get_b());
    }

    template <>
    void FileWriter::write<Matrix>(const Matrix &m) {
        stream_ << m.rows() << " " << m.cols() << std::endl;
        for (Index i = 0; i < m.rows(); ++i) {
            for (Index j = 0; j < m.cols(); ++j) {
                stream_ << m(i, j) << " ";
            }
            stream_ << std::endl;
        }
    }

    template <>
    void FileWriter::write<Vector>(const Vector &m) {
        stream_ << m.size() << std::endl;
        for (Index i = 0; i < m.size(); ++i) {
            stream_ << m(i) << std::endl;
        }
        stream_ << std::endl;
    }
} // namespace CNetworks

