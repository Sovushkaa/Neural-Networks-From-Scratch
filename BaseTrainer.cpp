#include "BaseTrainer.h"
#include <iostream>
#include <iomanip>

namespace CNetworks {
  BaseTrainer::BaseTrainer(DataLoader &&loader, LossFunction &&loss, Epoch_number epoch_number, Batch_size batch_size)
    : loader_(std::move(loader)), loss_(std::move(loss)), epoch_number_(epoch_number), batch_size_(batch_size) {
    loader_.SetBatchSize(batch_size);
  }

  BaseTrainer::Metrics BaseTrainer::ComputeMetrics(const Network &net) const {
    Matrix z = net.Evaluate(loader_.GetTestX());
    Matrix y = loader_.GetTestY();
    return {
      .average_mistake = loss_.AverageMistake(z, y),
      .success_percent = loss_.SuccessPercent(z, y)
    };
  }

  void BaseTrainer::PrintMetrics(const Metrics &metrics) const {
    std::cout << "Test mistake: " << std::fixed << std::setprecision(10) << metrics.average_mistake << std::endl;
    std::cout << "Test success percent: " << std::fixed << std::setprecision(5) << metrics.success_percent << "%" <<
        std::endl;
  }
}
