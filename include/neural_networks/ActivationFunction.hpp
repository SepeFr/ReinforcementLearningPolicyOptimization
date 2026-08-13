#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

/**
 * @defgroup neural_network_api Neural Networks
 * @brief Feed-forward layers, policies, parameter layouts, factories, and serialization.
 * @{
 */

#include <eigen3/Eigen/Core>
#include <stdexcept>
#include "ActivationType.hpp"
/** @brief Applies an ActivationType to every component of a vector. */
class ActivationFunction
{
  public:
  /**
   * @brief Evaluates the selected activation element by element.
   * @param[in] type Activation to evaluate.
   * @param[in] input Vector of pre-activation values.
   * @return Newly allocated vector with the same dimension as @p input.
   * @throws std::invalid_argument If @p type falls outside the supported enum values.
   */
  static __attribute__( ( always_inline ) ) Eigen::VectorXd apply( ActivationType type,
                                                                   Eigen::Ref< const Eigen::VectorXd > input )
  {
    switch ( type )
    {
      case ActivationType::Linear:
        return input;
      case ActivationType::ReLu:
        return input.array().max( 0 ).matrix();
      case ActivationType::Tanh:
        return input.array().tanh().matrix();
      case ActivationType::Sigmoid:
        return ( 1.0 / ( 1.0 + ( -input.array() ).exp() ) ).matrix();
      default:
        throw std::invalid_argument( "Unsupported ActivationType" );
    }
  }
};

/** @} */

#endif // !ACTIVATION_FUNCTION_H
