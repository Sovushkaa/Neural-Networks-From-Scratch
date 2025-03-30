#pragma once
#include "Network.h"

namespace CNetworks {
    class NetworkBuilder {
    public:
        NetworkBuilder(In input_size);
        NetworkBuilder &add_layer(Out output_size, ActivationFunction function);
        Network extract();

    private:
        Index last_layer_size_;
        Network network_;
    };
}

