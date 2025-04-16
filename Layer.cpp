#include "Layer.h"
#include "Random.h"
#include <iostream>

namespace CNetworks {
    Layer::Layer(In in_size, Out out_size, ActivationFunction f, Random &rnd)
        : A_(rnd.normalMatrix(out_size, in_size)), b_(rnd.normalVector(out_size)), f_(std::move(f)) {
    }

    Layer::Layer(Matrix &&A, Vector &&b, ActivationFunction f)
        : A_(std::move(A)), b_(std::move(b)), f_(std::move(f)) {
    }

    Index Layer::GetInputSize() const { return A_.cols(); }
    Index Layer::GetOutputSize() const { return A_.rows(); }
    const Matrix &Layer::Get_A() const { return A_; }
    const Vector &Layer::Get_b() const { return b_; }
    Matrix Layer::Zero_A() const { return Matrix::Zero(A_.rows(), A_.cols()); };
    Matrix Layer::Zero_b() const { return Vector::Zero(A_.rows()); };

    Matrix Layer::Evaluate(const Matrix &x) const {
        return f_.eval0((A_ * x).colwise() + b_);
    }

    Matrix Layer::Gradient_A(const Matrix &x, const Matrix &u) const {
        Matrix grad = Matrix::Zero(GetOutputSize(), GetInputSize());
        for (Index j = 0; j < x.cols(); ++j) {
            grad += f_.eval1(A_ * x.col(j) + b_).asDiagonal() * u.row(j).transpose() *
                    x.col(j).transpose();
        }
        return grad / x.cols();
    }

    Vector Layer::Gradient_b(const Matrix &x, const Matrix &u) const {
        Vector grad = Row::Zero(GetOutputSize());
        for (Index j = 0; j < x.cols(); ++j) {
            grad += f_.eval1(A_ * x.col(j) + b_).asDiagonal() * u.row(j).transpose();
        }
        return grad / x.cols();
    }

    Matrix Layer::Push(const Matrix &x, const Matrix &u) const {
        Matrix u_new = Matrix::Zero(x.cols(), GetInputSize());
        for (Index j = 0; j < x.cols(); ++j) {
            u_new.row(j) = u.row(j) * f_.eval1(A_ * x.col(j) + b_).asDiagonal() * A_;
        }
        return u_new;
    }

    void Layer::Update(const Matrix &update_A, const Vector &update_b) {
        A_ -= update_A;
        b_ -= update_b;
    }
} // namespace CNetworks
