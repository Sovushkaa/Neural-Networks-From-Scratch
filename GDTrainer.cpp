#include "GDTrainer.h"

namespace CNetworks {
    GDTrainer::GDTrainer(DataLoader &&loader, LossFunction &&loss,
                         Epoch_number epoch_number,
                         double learning_rate)
        : SGDTrainer(
            std::move(DataLoader()), std::move(loss), epoch_number,
            Batch_size{loader.GetTrainSize()}, learning_rate) {
        loader.SetBatchSize(batch_size_);
        this->loader_ = std::move(loader);
    }
}
