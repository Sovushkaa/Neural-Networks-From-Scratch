#include "NetworkBuilder.h"

namespace CNetworks {
    NetworkBuilder::NetworkBuilder(In input_size) : last_layer_size_(input_size) {}

    NetworkBuilder &NetworkBuilder::add_layer(Out output_size,
                                              ActivationFunction function) {
        network_.addLayer(Out{output_size}, In{last_layer_size_},
                          std::move(function));
        last_layer_size_ = output_size;
        return *this;
    }

    Network NetworkBuilder::extract() { return std::move(network_); }
}

