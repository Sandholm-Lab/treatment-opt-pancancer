#ifndef SRC_SAMPLER_TRUNCATED_NORMAL_SAMPLER_H_
#define SRC_SAMPLER_TRUNCATED_NORMAL_SAMPLER_H_

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include "HmcSampler.h"

using Vector = Eigen::VectorXd;
using DenseMatrix = Eigen::MatrixXd;
namespace py = pybind11;

/**
 * This object can sample from a normal distribution N(mu, cov) truncated with
 * linear constraints.
 *
 * The dimension of the space is inferred automatically from the size of mu.
 *
 *
 * ## Note
 *
 * We require `mu` to be strictly feasible.
 */
class TruncatedNormalSampler {
 public:
  TruncatedNormalSampler(const Vector mu, const DenseMatrix cov,
                         const size_t seed);

  /**
   * Adds a linear constraint of the form `coeff^T x >= rhs`.
   */
  void AddLinearConstraint(const Vector coeff, const double rhs);

  Vector Sample();
  Vector SampleWithBurnIn(const size_t burn_in);

 protected:
  /**
   * The internal sampler can only sample from a standard normal. In order to
   * bridge this gap, we perform a change of basis. In particular, instad of
   * sampling a vector x from a distribution with log density
   *
   * log p(x) = (x - mu)^T cov^-1 (x - mu) + k,
   *
   * we sample a vector y from the standard distribution
   *
   * log q(y) = y^T y + h,
   *
   * and then map y to x = cov^(1/2) y + mu. Of course, the linear constraints
   * (which are expressed in terms of x) need to be rewritten in terms of y
   * before being passed on to the internal `hmc_sampler_`.
   */
  DenseMatrix cov_sqrt_;
  Vector mu_;
  HmcSampler hmc_sampler_;
};

PYBIND11_MODULE(sampler, m) {
  m.doc() = "Sampler for truncated normal distributions";
  py::class_<TruncatedNormalSampler>(m, "TruncatedNormalSampler")
      .def(py::init<const Vector&, const DenseMatrix&, const size_t>())
      .def("add_linear_constraint",
           &TruncatedNormalSampler::AddLinearConstraint)
      .def("sample", &TruncatedNormalSampler::Sample)
      .def("sample_with_burn_in", &TruncatedNormalSampler::SampleWithBurnIn);
}

#endif  // SRC_SAMPLER_TRUNCATED_NORMAL_SAMPLER_H_
