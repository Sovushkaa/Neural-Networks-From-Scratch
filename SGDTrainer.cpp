#include "SGDTrainer.h"
#include <chrono>
#include <iomanip>
#include <iostream>

namespace CNetworks {
  SGDTrainer::SGDTrainer(DataLoader &&loader, LossFunction &&loss,
                         Epoch_number epoch_number, Batch_size batch_size,
                         double learning_rate)
    : BaseTrainer(std::move(loader), std::move(loss), epoch_number, batch_size),
      learning_rate_(learning_rate) {
  }

  Network SGDTrainer::Train(Network &&net) {
    for (Index it = 0; it < epoch_number_; ++it) {
      std::cout << "Epoch " << it + 1 << std::endl;
      auto metrics = ComputeMetrics(net);
      PrintMetrics(metrics);
      loader_.ShuffleTrainData();
      TrainOneEpoch(net.layers_);
    }
    return net;
  }

  void SGDTrainer::TrainOneEpoch(std::vector<Layer> &layers) {
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
        layers[k].Update(grad_A * learning_rate_, grad_b * learning_rate_);
      }
    }
  }
} // namespace CNetworks
