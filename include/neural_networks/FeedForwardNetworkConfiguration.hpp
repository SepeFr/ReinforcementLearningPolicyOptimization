#ifndef FEED_FORWARD_NETWORK_CONFIGURATION_H
#define FEED_FORWARD_NETWORK_CONFIGURATION_H

/** @addtogroup neural_network_api
 * @{ */

#include <cstddef>
#include <vector>
#include "ActivationType.hpp"
/** @brief Defines the dimensions and activations of a feed-forward network. */
struct FeedForwardNetworkConfiguration
{
  public:
  /**
   * @brief Computes the serialized parameter count with checked arithmetic.
   *
   * Each adjacent pair of layer widths contributes their product in weights;
   * each non-input width additionally contributes biases when use_bias is set.
   *
   * @param[in] configuration Network dimensions and bias setting to inspect.
   * @return Total number of scalar parameters.
   * @throws std::overflow_error If a dimension is not representable as
   * Eigen::Index or count arithmetic overflows.
   */
  static std::size_t parameterCountFromConfiguration( const FeedForwardNetworkConfiguration &configuration );

  std::size_t input_size;                 ///< Number of components expected by the first layer; must be positive.
  std::vector< std::size_t > hidden_layers; ///< Hidden widths in propagation order; every width must be positive.
  std::size_t output_size;                ///< Number of components produced by the final layer; must be positive.

  ActivationType hidden_activation = ActivationType::Sigmoid; ///< Activation shared by all hidden layers.
  ActivationType output_activation = hidden_activation; ///< Activation applied by the output layer.

  bool use_bias = true; ///< Enables one bias per non-input unit in every layer.
};

/** @} */

#endif // !FEED_FORWARD_NETWORK_CONFIGURATION_H
