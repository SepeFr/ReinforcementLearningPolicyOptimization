#ifndef OPTIMIZER_SIMULATED_ANNEALING_COOLING_FUNCTION_CONFIGURATION_H
#define OPTIMIZER_SIMULATED_ANNEALING_COOLING_FUNCTION_CONFIGURATION_H

/** @addtogroup optimization_methods_api
 * @{ */

/** @file CoolingFunctionConfiguration.hpp @brief Simulated-annealing cooling configuration alternatives. */

#include <variant>

struct GeometricCoolingConfiguration;
struct LinearCoolingConfiguration;
struct LogarithmicCoolingConfiguration;
struct ReciprocalCoolingConfiguration;
struct ExponentialCoolingConfiguration;
struct InverseLogarithmicCoolingConfiguration;

/** @brief Variant selecting a simulated-annealing temperature schedule. */
using CoolingFunctionConfiguration =
  std::variant< GeometricCoolingConfiguration, LinearCoolingConfiguration, LogarithmicCoolingConfiguration,
                ReciprocalCoolingConfiguration, ExponentialCoolingConfiguration,
                InverseLogarithmicCoolingConfiguration >;

/** @brief Configures geometric cooling. */
struct GeometricCoolingConfiguration
{
  double alpha = 0.95; ///< Finite decay factor strictly between zero and one.
};

/** @brief Configures linearly decreasing cooling with a floor. */
struct LinearCoolingConfiguration
{
  double beta = 0.01; ///< Finite positive decrease per algorithmic time unit.
  double minimum_temperature = 0.0; ///< Finite non-negative schedule floor below the initial temperature.
};

/** @brief Configures \f$c/\log(k+d)\f$ cooling. */
struct LogarithmicCoolingConfiguration
{
  double c = 1.0; ///< Finite positive numerator.
  double d = 0.0; ///< Finite positive time offset after validation.
};

/** @brief Configures reciprocal cooling from the initial temperature. */
struct ReciprocalCoolingConfiguration
{
  double b = 0.01; ///< Finite positive denominator growth coefficient.
};

/** @brief Configures continuous exponential cooling. */
struct ExponentialCoolingConfiguration
{
  double beta = 0.01; ///< Finite positive exponential decay rate.
};

/** @brief Configures the dimension-based inverse-logarithmic schedule. */
struct InverseLogarithmicCoolingConfiguration
{
  double dimension = 2.0; ///< Finite effective dimension strictly greater than one.
};

/** @} */

#endif // !OPTIMIZER_SIMULATED_ANNEALING_COOLING_FUNCTION_CONFIGURATION_H
