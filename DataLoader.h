#pragma once
#include "ReadCSV.h"
#include "linalg.h"

namespace CNetworks {
    class DataLoader {
    public:
        DataLoader(const std::string &train_path, const std::string &test_path);

        const Matrix &GetTestX() const;

        const Matrix &GetTestY() const;

        int GetNumBatches() const;

        void SetBatchSize(int batch_size);

        void ShuffleTrainData();

        class BatchIterator {
        public:
            BatchIterator(const Matrix &x, const Matrix &y, int batch_idx,
                          int batch_size);

            std::pair<Matrix, Matrix> operator*() const;

            BatchIterator &operator++();

            bool operator!=(const BatchIterator &other) const;

        private:
            const Matrix &x_;
            const Matrix &y_;
            int batch_num_;
            int batch_size_;
        };

        struct BatchRange {
            BatchIterator begin() const;

            BatchIterator end() const;

            const DataLoader &loader;
        };

        BatchRange batches() const;

        std::pair<Matrix, Matrix> GetBatch(Index batch_num) const;

    private:
        Matrix x_test_;
        Matrix y_test_;
        Matrix x_train_;
        Matrix y_train_;
        int batch_size_ = 32;
    };
} // namespace CNetworks
