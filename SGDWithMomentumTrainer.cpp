#include "SGDWithMomentumTrainer.h"
#include <chrono>
#include <iomanip>
#include <iostream>

namespace CNetworks {
    SGDWithMomentumTrainer::SGDWithMomentumTrainer(DataLoader &&loader, LossFunction &&loss,
                                                   Epoch_number epoch_number, Batch_size batch_size,
                                                   double learning_rate, double momentum_term)
        : BaseTrainer(std::move(loader), std::move(loss), epoch_number, batch_size),
          learning_rate_(learning_rate), momentum_term_(momentum_term) {
    }

    Network SGDWithMomentumTrainer::Train(Network &&net) {
        for (const auto &layer: net.layers_) {
            updates_.push_back(Layer_inf(layer.Zero_A(), layer.Zero_b()));
        }
        for (Index it = 0; it < epoch_number_; ++it) {
            std::cout << "Epoch " << it + 1 << std::endl;
            auto metrics = ComputeMetrics(net);
            PrintMetrics(metrics);
            loader_.ShuffleTrainData();
            TrainOneEpoch(net.layers_);
        }
        return net;
    }

    void SGDWithMomentumTrainer::TrainOneEpoch(std::vector<Layer> &layers) {
        for (const auto &[x, y]: loader_.batches()) {
            std::vector<Matrix> x_i;
            x_i.reserve(layers.size() + 1);
            x_i.push_back(x);
            Matrix z = x;
            for (const auto &layer: layers) {
                z = layer.Evaluate(x_i.back());
                x_i.push_back(std::move(z));
            }
            Matrix u = loss_.Grad(x_i.back(), y);
            for (Index k = layers.size() - 1; k >= 0; --k) {
                Matrix grad_A = layers[k].Gradient_A(x_i[k], u);
                Vector grad_b = layers[k].Gradient_b(x_i[k], u);
                u = layers[k].Push(x_i[k], u);
                updates_[k].A = updates_[k].A * momentum_term_ + grad_A * learning_rate_;
                updates_[k].b = updates_[k].b * momentum_term_ + grad_b * learning_rate_;
                layers[k].Update(updates_[k].A, updates_[k].b);
            }
        }
    }
} // namespace CNetworks
