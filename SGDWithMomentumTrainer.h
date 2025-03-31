#pragma once
#include "BaseTrainer.h"

namespace CNetworks {
    class SGDWithMomentumTrainer : public BaseTrainer {
    public:
        SGDWithMomentumTrainer(DataLoader &&loader, LossFunction &&loss, Epoch_number epoch_number,
                               Batch_size batch_size, double learning_rate = 0.001, double momentum_term = 0.9);

        Network Train(Network &&net) override;

    private:
        void TrainOneEpoch(std::vector<Layer> &layers) override;

        std::vector<Layer_inf> updates_;
        double learning_rate_;
        double momentum_term_;
    }; // namespace CNetworks
} // namespace CNetworks
