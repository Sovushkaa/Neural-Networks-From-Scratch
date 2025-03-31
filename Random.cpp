#include "Random.h"

namespace CNetworks {
    Random::Random(int seed) : generator_(seed) {
    }

    Matrix Random::normalMatrix(Index rows, Index cols) {
        std::normal_distribution<double> dist(0.0, 1.0);
        Matrix ans(rows, cols);
        for (Index i = 0; i < rows; ++i) {
            for (Index j = 0; j < cols; ++j) {
                ans(i, j) = dist(generator_);
            }
        }
        return ans;
    }

    Vector Random::normalVector(Index rows) {
        std::normal_distribution<double> dist(0.0, 1.0);
        Vector ans(rows);
        for (Index i = 0; i < rows; ++i) {
            ans(i) = dist(generator_);
        }
        return ans;
    }

    PermutationMatrix Random::permutation(Index size) {
        Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(size, 0, size - 1);
        std::shuffle(ind.data(), ind.data() + size, generator_);
        return PermutationMatrix(ind);
    }
}
