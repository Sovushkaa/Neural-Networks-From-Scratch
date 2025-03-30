#include "SAGTrainer.h"
#include <chrono>
#include <iomanip>
#include <iostream>

namespace CNetworks {
SAGTrainer::SAGTrainer(DataLoader &&loader, LossFunction &&loss,
                       Epoch_number epoch_number, Batch_size batch_size,
                       double learning_rate)
    : loader_(std::move(loader)), loss_(std::move(loss)),
      epoch_number_(epoch_number), batch_size_(batch_size),
      learning_rate_(learning_rate) {
  loader_.SetBatchSize(batch_size);
}

Network SAGTrainer::Train(Network &&net) {
  for (Index it = 0; it < epoch_number_; ++it) {
    std::cout << "Epoch " << it + 1 << std::endl;
    auto metrics = ComputeMetrics(net);
    PrintMetrics(metrics);
    loader_.ShuffleTrainData();
    //TrainOneEpoch(net.layers_);
  }
  return net;
}

void SAGTrainer::TrainOneEpoch(std::vector<Layer> &layers) {
  for (const auto &[x, y] : loader_.batches()) {
    std::vector<Matrix> x_i;
    x_i.reserve(layers.size() + 1);
    x_i.push_back(x);
    Matrix z = x;
    for (const auto &layer : layers) {
      z = layer.Evaluate(x_i.back());
      x_i.push_back(std::move(z));
    }
    Matrix u = loss_.Grad(x_i.back(), y);
    for (Index k = layers.size() - 1; k >= 0; --k) {
      Matrix grad_A = layers[k].Gradient_A(x_i[k], u);
      Vector grad_b = layers[k].Gradient_b(x_i[k], u);
      u = layers[k].Push(x_i[k], u);
      layers[k].Update(learning_rate_, grad_A, grad_b);
    }
  }
}

SAGTrainer::Metrics SAGTrainer::ComputeMetrics(const Network &net) const {
  Matrix z = net.Evaluate(loader_.GetTestX());
  Matrix y = loader_.GetTestY();
  return {.average_mistake = loss_.AverageMistake(z, y),
          .success_percent = loss_.SuccessPercent(z, y)};
}

void SAGTrainer::PrintMetrics(const Metrics &metrics) const {
  std::cout << "Test mistake: " << std::fixed << std::setprecision(10)
            << metrics.average_mistake << std::endl;
  std::cout << "Test success percent: " << std::fixed << std::setprecision(5)
            << metrics.success_percent << "%" << std::endl;
}
} // namespace CNetworks
