#include "ActivationFunction.h"

namespace CNetworks {
    enum In : Index;
    enum Out : Index;

    class Layer {
    public:
        Layer(In in_size, Out out_size,
              ActivationFunction f = ActivationFunction::Sigmoid());
        Layer(Matrix &&A, Vector &&b,
              ActivationFunction f = ActivationFunction::Sigmoid());

        Index GetInputSize() const;
        Index GetOutputSize() const;
        const Matrix &Get_A() const;
        const Vector &Get_b() const;
        Matrix Zero_A() const;
        Matrix Zero_b() const;

        Matrix Evaluate(const Matrix &x) const;
        Matrix Gradient_A(const Matrix &x, const Matrix &u) const;
        Vector Gradient_b(const Matrix &x, const Matrix &u) const;
        Matrix Push(const Matrix &x, const Matrix &u) const;
        void Update(double c, const Matrix &grad_a, const Vector &grad_b);
        void AdamUpdate(const Matrix &update_A, const Vector &update_b);

    private:
        Matrix A_;
        Vector b_;
        ActivationFunction f_;
    };
} // namespace CNetworks