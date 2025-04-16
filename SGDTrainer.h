#pragma once
#include "BaseTrainer.h"

namespace CNetworks {
    class SGDTrainer : public BaseTrainer {
    public:
        SGDTrainer(DataLoader &&loader, LossFunction &&loss, Epoch_number epoch_number,
                   Batch_size batch_size, double learning_rate);

        Network Train(Network &&net);

    private:
        void TrainOneEpoch(std::vector<Layer> &layers);

        double learning_rate_;
    }; // namespace CNetworks
} // namespace CNetworks
