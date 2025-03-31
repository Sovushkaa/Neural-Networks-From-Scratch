#pragma once
#include "BaseTrainer.h"

namespace CNetworks {
    class AdamTrainer : public BaseTrainer {
    public:
        AdamTrainer(DataLoader &&loader, LossFunction &&loss,
                    Epoch_number epoch_number, Batch_size batch_size,
                    double alpha_ = 0.001, double beta1 = 0.9, double beta2 = 0.999,
                    double eps = 1e-8);

        Network Train(Network &&net) override;

    private:
        void TrainOneEpoch(std::vector<Layer> &layers);

        double alpha_;
        double beta1_;
        double beta2_;
        double eps_;
        std::vector<Layer_inf> m_;
        std::vector<Layer_inf> u_;
        Index timestep_ = 0;
    };
} // namespace CNetworks

