#pragma once
#include "Layer.h"
#include "LossFunction.h"
#include <iostream>

namespace CNetworks {
    class Network {
        friend class NetworkBuilder;
        friend class FileReader;
        friend class FileWriter;
        friend class GDTrainer;
        friend class SGDTrainer;
        friend class SGDWithMomentumTrainer;
        friend class AdamTrainer;

    public:
        Network(std::vector<Layer> &&layers);

        Index GetLayersCount() const;

        Matrix Evaluate(const Matrix &x) const;

        Matrix Evaluate(Matrix &&x) const;

    private:
        Network() = default;

        void addLayer(Out output_size, In input_size, ActivationFunction function);

        std::vector<Layer> layers_;
    };
} // namespace CNetworks

