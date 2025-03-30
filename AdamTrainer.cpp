#include "AdamTrainer.h"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace CNetworks {
AdamTrainer::AdamTrainer(DataLoader &&loader, LossFunction &&loss,
                         Epoch_number epoch_number, Batch_size batch_size,
                         double alpha_, double beta1, double beta2, double eps)
    : loader_(std::move(loader)), loss_(std::move(loss)),
      epoch_number_(epoch_number), batch_size_(batch_size), alpha_(alpha_),
      beta1_(beta1), beta2_(beta2), eps_(eps) {
  loader_.SetBatchSize(batch_size);
}

Network AdamTrainer::Train(Network &&net) {
  m_.reserve(net.GetLayersCount() + 1);
  u_.reserve(net.GetLayersCount() + 1);
  for (const auto &layer : net.layers_) {
    m_.push_back(Layer_inf(layer.Zero_A(), layer.Zero_b()));
    u_.push_back(Layer_inf(layer.Zero_A(), layer.Zero_b()));
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

void AdamTrainer::TrainOneEpoch(std::vector<Layer> &layers) {
  for (const auto &[x, y] : loader_.batches()) {
    ++timestep_;
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
      m_[k].A = m_[k].A * beta1_ + grad_A * (1 - beta1_);
      m_[k].b = m_[k].b * beta1_ + grad_b * (1 - beta1_);
      u_[k].A =
          u_[k].A * beta2_ + grad_A.array().square().matrix() * (1 - beta2_);
      u_[k].b =
          u_[k].b * beta2_ + grad_b.array().square().matrix() * (1 - beta2_);
      Matrix m_hat_A = m_[k].A / (1 - std::pow(beta1_, timestep_));
      Vector m_hat_b = m_[k].b / (1 - std::pow(beta1_, timestep_));
      Matrix u_hat_A = u_[k].A / (1 - std::pow(beta2_, timestep_));
      Vector u_hat_b = u_[k].b / (1 - std::pow(beta2_, timestep_));
      Matrix A_update =
          (alpha_ * m_hat_A.array() / ((u_hat_A.array().sqrt() + eps_)))
              .matrix();
      Vector b_update =
          (alpha_ * m_hat_b.array() / ((u_hat_b.array().sqrt() + eps_)))
              .matrix();
      layers[k].AdamUpdate(A_update, b_update);
    }
  }
}

AdamTrainer::Metrics AdamTrainer::ComputeMetrics(const Network &net) const {
  Matrix z = net.Evaluate(loader_.GetTestX());
  Matrix y = loader_.GetTestY();
  return {.average_mistake = loss_.AverageMistake(z, y),
          .success_percent = loss_.SuccessPercent(z, y)};
}

void AdamTrainer::PrintMetrics(const Metrics &metrics) const {
  std::cout << "Test mistake: " << std::fixed << std::setprecision(10)
            << metrics.average_mistake << std::endl;
  std::cout << "Test success percent: " << std::fixed << std::setprecision(5)
            << metrics.success_percent << "%" << std::endl;
}
} // namespace CNetworks*/

