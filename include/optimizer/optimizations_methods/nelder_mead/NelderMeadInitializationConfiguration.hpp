#ifndef NELDER_MEAD_INITIALIZATION_CONFIGURATION_H
#define NELDER_MEAD_INITIALIZATION_CONFIGURATION_H

/** @addtogroup optimization_methods_api
 * @{ */

/** @file NelderMeadInitializationConfiguration.hpp @brief Simplex initialization configuration alternatives. */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <variant>

/** @brief Selects an axis-aligned simplex around one generated center. */
struct ClassicalLocalSimplexConfiguration
{
};

/** @brief Selects a full-rank Latin hypercube simplex around one generated center. */
struct LatinHypercubeLocalSimplexConfiguration
{
  std::size_t maximum_attempts = 16; ///< Positive maximum designs tried before failure.
  std::size_t random_seed = 0; ///< Seed used by the local Latin hypercube sampler.
};

/** @brief Selects a full-rank Latin hypercube simplex inside a configured box. */
struct LatinHypercubeGlobalSimplexConfiguration
{
  Eigen::VectorXd lower_bound; ///< Finite lower design limits in problem-coordinate order.
  Eigen::VectorXd upper_bound; ///< Finite strict upper design limits in matching order.
  std::size_t maximum_attempts = 16; ///< Positive maximum designs tried before failure.
  std::size_t random_seed = 0; ///< Seed used by the global Latin hypercube sampler.
};

/** @brief Selects consecutive vectors from ProvidedInitializationVector. */
struct ProvidedSimplexConfiguration
{
};

/** @brief Variant selecting construction of the initial \f$n+1\f$ simplex vertices. */
using NelderMeadInitializationConfiguration =
  std::variant< ClassicalLocalSimplexConfiguration, LatinHypercubeLocalSimplexConfiguration,
                LatinHypercubeGlobalSimplexConfiguration, ProvidedSimplexConfiguration >;

/** @} */

#endif // !NELDER_MEAD_INITIALIZATION_CONFIGURATION_H
