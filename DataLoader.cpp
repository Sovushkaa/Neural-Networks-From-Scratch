#include "DataLoader.h"
#include <cmath>
#include <fstream>
#include <random>
#include <cassert>

namespace CNetworks {
  DataLoader::DataLoader(const std::string &train_path,
                         const std::string &test_path) {
    try {
      auto train_data = ReadCSV(train_path);
      auto test_data = ReadCSV(test_path);
      x_train_ = train_data.first;
      y_train_ = train_data.second;
      x_test_ = test_data.first;
      y_test_ = test_data.second;
    } catch (...) {
      x_train_ = Matrix();
      y_train_ = Matrix();
      x_test_ = Matrix();
      y_test_ = Matrix();
    }
  }

  const Matrix &DataLoader::GetTestX() const { return x_test_; }
  const Matrix &DataLoader::GetTestY() const { return y_test_; }

  Index DataLoader::GetTrainSize() const {
    return x_train_.cols();
  }


  Index DataLoader::GetNumBatches() const { return x_train_.cols() / batch_size_; }

  void DataLoader::SetBatchSize(int batch_size) { batch_size_ = batch_size; }

  void DataLoader::ShuffleTrainData(Random &rnd) {
    auto perm = rnd.permutation(x_test_.cols());
    x_test_ *= perm;
    y_test_ *= perm;
  }

  DataLoader::BatchIterator::BatchIterator(const Matrix &x, const Matrix &y,
                                           int batch_idx, int batch_size)
    : x_(x), y_(y), batch_num_(batch_idx), batch_size_(batch_size) {
  }

  std::pair<Matrix, Matrix> DataLoader::BatchIterator::operator*() const {
    int start_num = batch_num_ * batch_size_;
    int end_num = std::min<int>((batch_num_ + 1) * batch_size_,
                                static_cast<int>(x_.cols()));
    Matrix x_batch = x_.middleCols(start_num, end_num - start_num);
    Matrix y_batch = y_.middleCols(start_num, end_num - start_num);
    return {x_batch, y_batch};
  }

  DataLoader::BatchIterator &DataLoader::BatchIterator::operator++() {
    ++batch_num_;
    return *this;
  }

  bool DataLoader::BatchIterator::operator!=(const BatchIterator &other) const {
    return batch_num_ != other.batch_num_;
  }

  DataLoader::BatchIterator DataLoader::BatchRange::begin() const {
    return BatchIterator(loader.x_train_, loader.y_train_, 0, loader.batch_size_);
  }

  DataLoader::BatchIterator DataLoader::BatchRange::end() const {
    return BatchIterator(loader.x_train_, loader.y_train_, loader.GetNumBatches(),
                         loader.batch_size_);
  }

  DataLoader::BatchRange DataLoader::batches() const { return BatchRange{*this}; }

  std::pair<Matrix, Matrix> DataLoader::GetBatch(Index batch_num) const {
    assert(batch_num >= 0 && batch_num < GetNumBatches());
    int start_num = batch_num * batch_size_;
    int end_num = std::min<int>((batch_num + 1) * batch_size_, static_cast<int>(x_train_.cols()));
    Matrix x_batch = x_train_.middleCols(start_num, end_num - start_num);
    Matrix y_batch = y_train_.middleCols(start_num, end_num - start_num);
    return {x_batch, y_batch};
  }
} // namespace CNetworks
