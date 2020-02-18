#include "truncated_normal_sampler.h"

#include <Eigen/Eigenvalues>

TruncatedNormalSampler::TruncatedNormalSampler(const Vector mu,
                                               const DenseMatrix cov,
                                               const size_t seed)
    : mu_(mu), hmc_sampler_(mu_.size(), seed) {
  Eigen::SelfAdjointEigenSolver<DenseMatrix> solver(cov);
  cov_sqrt_ = solver.operatorSqrt();

  // By hypothesis `mu` is strictly feasible.
  hmc_sampler_.setInitialValue(Vector::Zero(mu_.size()));
}

void TruncatedNormalSampler::AddLinearConstraint(const Vector coeff,
                                                 const double rhs) {
  // We perform the change of variable x = cov^(1/2) y + mu, so that the
  // constraint
  //    c^T x >= rhs
  // becomes
  //   (cov^(1/2) c)^T y >= rhs - c^T mu.
  const Vector new_c = cov_sqrt_ * coeff;
  const double new_rhs = rhs - coeff.dot(mu_);

  // Finally, we flip the sign of `new_rhs` since the internal sampler expects
  // linear constraints to be in the form `a^T x + b >= 0`.
  hmc_sampler_.addLinearConstraint(new_c, -new_rhs);
}

Vector TruncatedNormalSampler::Sample() {
  const Vector y = hmc_sampler_.sampleNext(/* returnTrace = */ false);
  return cov_sqrt_ * y + mu_;
}

Vector TruncatedNormalSampler::SampleWithBurnIn(const size_t burn_in) {
  for (size_t i = 0; i < burn_in; ++i) {
    hmc_sampler_.sampleNext(/* returnTrace = */ false);
  }
  return Sample();
}
