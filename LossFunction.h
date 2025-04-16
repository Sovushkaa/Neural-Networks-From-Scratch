#pragma once
#include "linalg.h"

namespace CNetworks {
    class LossFunction {
    public:
        Matrix Grad(const Matrix &z, const Matrix &y) const;
        double AverageMistake(const Matrix &z, const Matrix &y) const;
        double SuccessPercent(const Matrix &z, const Matrix &y) const;

        static LossFunction Mse();
        static LossFunction CrossEntropy();

    private:
        using Func1 = std::function<double(const Matrix &, const Matrix &)>;
        using Func2 = std::function<Matrix(const Matrix &, const Matrix &)>;
        LossFunction(Func1 loss, Func2 grad);
        int convert_to_number(Vector y) const;

        Func1 func0_;
        Func2 func1_;
    };
} // namespace CNetworks

