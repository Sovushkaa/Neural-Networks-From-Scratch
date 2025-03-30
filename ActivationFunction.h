#pragma once
#include "linalg.h"

namespace CNetworks {
    class ActivationFunction {
        using Function = std::function<double(double)>;
        using Derivative = std::function<Matrix(const Vector &)>;

    public:
        ActivationFunction(Function f0, Function f1, Derivative d);
        Matrix eval0(const Matrix &x) const;
        Vector eval1(const Vector &x) const;
        Matrix Jacoby(const Vector &x) const;
        static ActivationFunction Sigmoid();
        static ActivationFunction ReLU();
        static ActivationFunction Softmax();

    private:
        Function f0_;
        Function f1_;
        Derivative jacoby_;
    };
} // namespace CNetworks
