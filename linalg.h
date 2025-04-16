#pragma once
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace CNetworks {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Row = Eigen::RowVectorXd;
    using Index = Eigen::Index;
    using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic>;
} // namespace CNetworks
