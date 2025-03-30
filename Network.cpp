#include "Network.h"
#include <fstream>

namespace CNetworks {
    Network::Network(std::vector<Layer> &&layers) : layers_(std::move(layers)) {}

    void Network::addLayer(Out output_size, In input_size,
                           ActivationFunction function) {
        layers_.emplace_back(input_size, output_size, function);
    }

    Index Network::GetLayersCount() const { return layers_.size(); }

    Matrix Network::Evaluate(const Matrix &x) const {
        Matrix arg = x;
        return Evaluate(std::move(arg));
    }

    Matrix Network::Evaluate(Matrix &&x) const {
        Matrix result = std::move(x);
        for (const auto &layer : layers_) {
            result = layer.Evaluate(result);
        }
        return result;
    }
} // namespace CNetworks

