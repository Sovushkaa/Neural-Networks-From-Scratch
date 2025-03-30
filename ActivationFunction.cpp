#include "ActivationFunction.h"
#include <cmath>

namespace CNetworks {
    ActivationFunction::ActivationFunction(Function f0, Function f1, Derivative d)
        : f0_(std::move(f0)), f1_(std::move(f1)), jacoby_(std::move(d)) {}

    Matrix ActivationFunction::eval0(const Matrix &x) const {
        return x.unaryExpr(f0_);
    }

    Vector ActivationFunction::eval1(const Vector &x) const {
        return x.unaryExpr(f1_);
    }

    Matrix ActivationFunction::Jacoby(const Vector &x) const {
        return jacoby_(x);
    }

    ActivationFunction ActivationFunction::Sigmoid() {
        return ActivationFunction(
            [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
            [](double x) { return 1.0 / (std::exp(x) + std::exp(-x) + 2); },
            [](const Vector& x) {
              Vector result = x.unaryExpr([](double val) {
                return 1.0 / (std::exp(val) + std::exp(-val) + 2);
              });
              return result.asDiagonal();
            });
    }

    ActivationFunction ActivationFunction::ReLU() {
        return ActivationFunction([](double x) { return x > 0.0 ? x : 0.0; },
                                  [](double x) { return x > 0.0 ? 1.0 : 0.0; },
                                  [](Vector x) {
                                    Vector result = x.unaryExpr([](double val) {
                                      return val > 0.0 ? 1.0 : 0.0;
                                    });
                                    return result.asDiagonal();
                                  });
    }

} // namespace CNetworks

