#pragma once
#include "DataLoader.h"
#include "Network.h"

namespace CNetworks {
    enum Epoch_number : Index;
    enum Batch_size : Index;

    class BaseTrainer {
    protected:
        BaseTrainer(DataLoader &&loader, LossFunction &&loss, Epoch_number epoch_number, Batch_size batch_size);

        struct Layer_inf {
            Matrix A;
            Vector b;

            Layer_inf(Matrix &&A, Vector &&b) : A(std::move(A)), b(std::move(b)) {
            }
        };

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
    }; // namespace CNetworks
} // namespace CNetworks

