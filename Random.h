#pragma once
#include "linalg.h"
#include <random>

namespace CNetworks {
    class Random {
        //using Generator = Eigen::Rand::Vmt19937_64; не получилось работать с перестановками
        using Generator = std::mt19937;

    public:
        static Random &global() {
            static Random rnd(k_default_seed_);
            return rnd;
        }

        Random(int seed);

        Matrix normalMatrix(Index rows, Index cols);

        Vector normalVector(Index rows);

        Eigen::PermutationMatrix<Eigen::Dynamic> permutation(Index size);

    private:
        static constexpr int k_default_seed_ = 42;
        Generator generator_{k_default_seed_};
    };
} // namespace CNetworks
