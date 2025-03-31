#pragma once
#include "SGDTrainer.h"

namespace CNetworks {
    class GDTrainer : public SGDTrainer {
    public:
        GDTrainer(DataLoader &&loader, LossFunction &&loss, Epoch_number epoch_number, double learning_rate);
    };
} // namespace CNetworks
