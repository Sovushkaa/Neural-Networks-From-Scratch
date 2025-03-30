#pragma once

#include "DataLoader.h"
#include "LossFunction.h"
#include "Network.h"

namespace CNetworks {
    enum Epoch_number : Index;
    enum Batch_size : Index;
    class AdamTrainer {
    public:
        AdamTrainer(DataLoader &&loader, LossFunction &&loss,
                    Epoch_number epoch_number, Batch_size batch_size,
                    double alpha_ = 0.001, double beta1 = 0.9, double beta2 = 0.999,
                    double eps = 1e-8);

        Network Train(Network &&net);

        struct Layer_inf {
            Matrix A;
            Vector b;
            Layer_inf(Matrix &&A, Vector &&b) : A(std::move(A)), b(std::move(b)) {}
        };

    private:
        void TrainOneEpoch(std::vector<Layer> &layers);

        struct Metrics {
            double average_mistake;
            double success_percent;
        };

        Metrics ComputeMetrics(const Network &net) const;
        void PrintMetrics(const Metrics &metrics) const;

        DataLoader loader_;
        LossFunction loss_;
        Index epoch_number_;
        Index batch_size_;
        double alpha_;
        double beta1_;
        double beta2_;
        double eps_;
        std::vector<Layer_inf> m_;
        std::vector<Layer_inf> u_;
        Index timestep_ = 0;
    };
} // namespace CNetworks

