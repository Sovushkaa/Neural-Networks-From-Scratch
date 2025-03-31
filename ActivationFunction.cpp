#include "ActivationFunction.h"
#include <cmath>

namespace CNetworks {
    ActivationFunction::ActivationFunction(Function f0, Function f1)
        : f0_(std::move(f0)), f1_(std::move(f1)) {
    }

    Matrix ActivationFunction::eval0(const Matrix &x) const {
        return x.unaryExpr(f0_);
    }

    Vector ActivationFunction::eval1(const Vector &x) const {
        return x.unaryExpr(f1_);
    }

    ActivationFunction ActivationFunction::Sigmoid() {
        return ActivationFunction(
            [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
            [](double x) { return 1.0 / (std::exp(x) + std::exp(-x) + 2); });
    }

    ActivationFunction ActivationFunction::ReLU() {
        return ActivationFunction([](double x) { return x > 0.0 ? x : 0.0; },
                                  [](double x) { return x > 0.0 ? 1.0 : 0.0; });
    }
} // namespace CNetworks

