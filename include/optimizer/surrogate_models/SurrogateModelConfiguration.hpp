#ifndef SURROGATE_MODEL_CONFIGURATION_H
#define SURROGATE_MODEL_CONFIGURATION_H

/** @addtogroup optimization_methods_api
 * @{ */

/** @file SurrogateModelConfiguration.hpp @brief Local quadratic and radial-basis surrogate settings. */

#include <cstddef>
#include <variant>

/** @brief Configures local quadratic regression in mesh-scaled coordinates. */
struct LocalQuadraticSurrogateConfiguration
{
  double curvature_regularization = 1.0e-6; ///< Finite non-negative squared-Frobenius curvature penalty.
  double local_radius_multiplier = 4.0; ///< Finite positive multiplier applied to the supplied mesh scale.
};

/** @brief Radial basis applied to scaled Euclidean distance. */
enum class RBFFunction
{
  Cubic, ///< \f$\phi(r)=r^3\f$.
  Linear ///< \f$\phi(r)=r\f$.
};

/** @brief Configures local RBF interpolation with an affine polynomial tail. */
struct RBFSurrogateConfiguration
{
  RBFFunction function = RBFFunction::Cubic; ///< Supported radial kernel.
  double regularization = 0.0; ///< Finite non-negative value added to the radial matrix diagonal.
  double local_radius_multiplier = 4.0; ///< Finite positive multiplier applied to the supplied mesh scale.
  std::size_t maximum_coreset_size = 512; ///< Positive cap on selected local observations.
};

/** @brief Variant selecting a regression model for surrogate GPS search. */
using SurrogateModelConfiguration = std::variant< LocalQuadraticSurrogateConfiguration, RBFSurrogateConfiguration >;

/** @} */

#endif // !SURROGATE_MODEL_CONFIGURATION_H
