#pragma once
#include "Network.h"
#include <filesystem>
#include <fstream>

namespace CNetworks {
    class FileWriter {
    public:
        FileWriter(const std::filesystem::path &filename);
        template <class Data>
        void write(const Data &data);

    private:
        std::ofstream stream_;
    };

    template <>
    void FileWriter::write<Network>(const Network &data);
    template <>
    void FileWriter::write<Layer>(const Layer &data);
    template <>
    void FileWriter::write<Matrix>(const Matrix &data);
    template <>
    void FileWriter::write<Vector>(const Vector &data);
} // namespace CNetworks

