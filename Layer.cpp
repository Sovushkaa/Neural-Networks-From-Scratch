#include "Layer.h"
#include <iostream>
#include <random>

namespace CNetworks {
Layer::Layer(In in_size, Out out_size, ActivationFunction f)
    : f_(std::move(f)) {
  std::mt19937 gen(42);
  std::normal_distribution<double> dis(0.0, 1.0);
  A_ = Matrix(out_size, in_size);
  b_ = Matrix(out_size, 1);
  for (int i = 0; i < out_size; ++i) {
    for (int j = 0; j < in_size; ++j) {
      A_(i, j) = dis(gen);
    }
    b_(i, 0) = dis(gen);
  }
}

Layer::Layer(Matrix &&A, Vector &&b, ActivationFunction f)
    : A_(std::move(A)), b_(std::move(b)), f_(std::move(f)) {}

Index Layer::GetInputSize() const { return A_.cols(); }
Index Layer::GetOutputSize() const { return A_.rows(); }
const Matrix &Layer::Get_A() const { return A_; }
const Vector &Layer::Get_b() const { return b_; }
Matrix Layer::Zero_A() const { return Matrix::Zero(A_.rows(), A_.cols()); };
Matrix Layer::Zero_b() const { return Vector::Zero(A_.rows()); };

Matrix Layer::Evaluate(const Matrix &x) const {
  return f_.eval0((A_ * x).colwise() + b_);
  // return f_.eval0(A_ * x + b_.replicate(1, x.cols()));
}

Matrix Layer::Gradient_A(const Matrix &x, const Matrix &u) const {
  Matrix grad = Matrix::Zero(GetOutputSize(), GetInputSize());
  for (Index j = 0; j < x.cols(); ++j) {
    grad += f_.eval1(A_ * x.col(j) + b_).asDiagonal() * u.row(j).transpose() *
            x.col(j).transpose();
  }
  return grad / x.cols();

  // return f_.eval1(A_ * x + b_).asDiagonal() * u.transpose() * x.transpose();
  //  return f_.Jacoby(A_ * x + b_) * u.transpose() * x.transpose();
  //  почему-то когда заменяю строки с eval1 на строчки с Jacaby
  //  результаты с фикс сидом начинают отличаться, становятся хуже,
}

Vector Layer::Gradient_b(const Matrix &x, const Matrix &u) const {
  Vector grad = Row::Zero(GetOutputSize());
  for (Index j = 0; j < x.cols(); ++j) {
    grad += f_.eval1(A_ * x.col(j) + b_).asDiagonal() * u.row(j).transpose();
  }
  return grad / x.cols();

  // return f_.eval1(A_ * x + b_).asDiagonal() * u.transpose();
  //  return f_.Jacoby(A_ * x + b_) * u.transpose();
}

Matrix Layer::Push(const Matrix &x, const Matrix &u) const {
  Matrix u_new = Matrix::Zero(x.cols(), GetInputSize());
  for (Index j = 0; j < x.cols(); ++j) {
    u_new.row(j) = u.row(j) * f_.eval1(A_ * x.col(j) + b_).asDiagonal() * A_;
  }
  return u_new;
  // return u * f_.eval1(A_ * x + b_).asDiagonal() * A_;
  //  return u * f_.Jacoby(A_ * x + b_) * A_;
}

void Layer::Update(double c, const Matrix &grad_a, const Vector &grad_b) {
  A_ -= grad_a * c;
  b_ -= grad_b * c;
}

void Layer::AdamUpdate(const Matrix &update_A, const Vector &update_b) {
  A_ -= update_A;
  b_ -= update_b;
}

} // namespace CNetworks

