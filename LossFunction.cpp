#include "LossFunction.h"
#include <cassert>
#include <iostream>

namespace CNetworks {
    Matrix LossFunction::Grad(const Matrix &z, const Matrix &y) const {
        assert(func1_);
        assert(z.rows() == y.rows() && z.cols() == y.cols());
        Matrix grad = func1_(z, y);
        return grad;
    }

    double LossFunction::AverageMistake(const Matrix &z, const Matrix &y) const {
        assert(z.rows() == y.rows() && z.cols() == y.cols() && "Not the same sizes");
        if (!z.cols()) {
            return 0;
        }
        return func0_(z, y) / z.cols();
    }

    double LossFunction::SuccessPercent(const Matrix &z, const Matrix &y) const {
        assert(z.rows() == y.rows() && z.cols() == y.cols() && "Not the same sizes");
        assert(z.cols());
        if (!z.cols()) {
            return 0;
        }
        int right_guess = 0;
        for (Index i = 0; i < z.cols(); ++i) {
            int predicted_ans = convert_to_number(z.col(i));
            int correct_ans = convert_to_number(y.col(i));
            if (predicted_ans == correct_ans) {
                ++right_guess;
            }
        }
        return (double(right_guess) / z.cols()) * 100;
    }

    LossFunction LossFunction::Mse() {
        return LossFunction(
            [](const Matrix &z, const Matrix &y) {
              return ((z - y).colwise().squaredNorm()).sum();
            },
            [](const Matrix &z, const Matrix &y) { return 2 * (z - y).transpose(); });
    }

    LossFunction LossFunction::CrossEntropy() {
        return LossFunction(
            [](const Matrix &z, const Matrix &y) {
              return -(y.array() * z.array().log()).sum();
            },
            [](const Matrix &z, const Matrix &y) {
              return -(y.array() / z.array()).matrix();
            });
    }

    LossFunction::LossFunction(Func1 loss, Func2 grad)
        : func0_(std::move(loss)), func1_(std::move(grad)) {}

    int LossFunction::convert_to_number(Vector y) const {
        assert(y.size() == 10);
        int ans = 0;
        for (Index i = 0; i < 10; ++i) {
            if (y(i) > y(ans)) {
                ans = i;
            }
        }
        return ans;
    }
} // namespace CNetworks

