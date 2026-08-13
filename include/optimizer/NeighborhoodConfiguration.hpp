#ifndef NEIGHBORHOOD_CONFIGURATION_H
#define NEIGHBORHOOD_CONFIGURATION_H

/** @addtogroup optimizer_core_api
 * @{ */

/** @file NeighborhoodConfiguration.hpp @brief Local displacement strategy configuration alternatives. */

#include <variant>

struct GaussianNeighborhoodConfiguration;
struct UniformNeighborhoodConfiguration;
struct CoordinateNeighborhoodConfiguration;
struct LatinHypercubeUniformNeighborhoodConfiguration;
struct LatinHypercubeGaussianNeighborhoodConfiguration;

/** @brief Variant selecting a perturbation strategy for local methods. */
using NeighborhoodConfiguration = std::variant< GaussianNeighborhoodConfiguration, UniformNeighborhoodConfiguration,
                                                CoordinateNeighborhoodConfiguration,
                                                LatinHypercubeUniformNeighborhoodConfiguration,
                                                LatinHypercubeGaussianNeighborhoodConfiguration >;

/** @brief Configures independent normally distributed coordinate displacements. */
struct GaussianNeighborhoodConfiguration
{
  double perturbation_mean = 0.0; ///< Mean displacement added to every coordinate.
  double standard_deviation = 1.0; ///< Finite positive displacement standard deviation.
};

/** @brief Configures independent uniform coordinate displacements. */
struct UniformNeighborhoodConfiguration
{
  double lower_bound; ///< Inclusive lower displacement endpoint.
  double upper_bound; ///< Inclusive upper displacement endpoint; at least lower_bound.
};


/** @brief Configures uniform displacements applied to randomly selected coordinates. */
struct CoordinateNeighborhoodConfiguration
{
  double lower_bound; ///< Inclusive lower displacement endpoint.
  double upper_bound; ///< Inclusive upper displacement endpoint; at least lower_bound.
};

/** @brief Configures a stratified uniform displacement batch. */
struct LatinHypercubeUniformNeighborhoodConfiguration
{
  double lower_bound = -1.0; ///< Inclusive lower displacement endpoint.
  double upper_bound = 1.0; ///< Inclusive upper displacement endpoint; at least lower_bound.
};

/** @brief Configures a stratified Gaussian displacement batch. */
struct LatinHypercubeGaussianNeighborhoodConfiguration
{
  double perturbation_mean = 0.0; ///< Mean displacement after inverse-normal transformation.
  double standard_deviation = 1.0; ///< Finite positive displacement standard deviation.
};


/** @} */

#endif // !NEIGHBORHOOD_CONFIGURATION_H
