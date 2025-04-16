#pragma once
#include "linalg.h"

namespace CNetworks {
    class ActivationFunction {
        using Function = std::function<double(double)>;

    public:
        ActivationFunction(Function f0, Function f1);

        Matrix eval0(const Matrix &x) const;

        Vector eval1(const Vector &x) const;

        static ActivationFunction Sigmoid();

        static ActivationFunction ReLU();

    private:
        Function f0_;
        Function f1_;
    };
} // namespace CNetworks
