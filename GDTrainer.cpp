#include "GDTrainer.h"

namespace CNetworks {
    GDTrainer::GDTrainer(DataLoader &&loader, LossFunction &&loss,
                         Epoch_number epoch_number, double learning_rate)
        : SGDTrainer(std::move(loader), std::move(loss), epoch_number, Batch_size{1}, learning_rate) {
    }
}
