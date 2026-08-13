#ifndef FEED_FORWARD_NETWORK_H
#define FEED_FORWARD_NETWORK_H

/** @addtogroup neural_network_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <vector>
#include "DenseLayer.hpp"
#include "FeedForwardNetworkConfiguration.hpp"
#include "ParametersLayout.hpp"
/**
 * @brief Owns and evaluates an ordered stack of DenseLayer objects.
 *
 * Hidden layers are created in configuration order, followed by one output
 * layer. The canonical parameter vector concatenates each layer's
 * column-major weights and biases in that same order.
 *
 * @see neural_parameter_mapping_chapter
 */
class FeedForwardNetwork
{
  public:
  /**
   * @brief Builds a zero-initialized network from a copied configuration.
   * @param[in] configuration Dimensions, activations, and bias setting.
   * @throws std::invalid_argument If an input, hidden, or output width is zero.
   * @throws std::overflow_error If dimensions or parameter counts exceed the
   * ranges supported by size_t or Eigen::Index.
   */
  explicit FeedForwardNetwork( const FeedForwardNetworkConfiguration &configuration );

  /**
   * @brief Propagates an input through all layers in order.
   * @param[in] input Vector whose length equals the configured input_size.
   * @return Output-layer activation vector.
   * @throws std::invalid_argument If the input length is invalid.
   */
  Eigen::VectorXd forward( Eigen::Ref< const Eigen::VectorXd > input ) const;

  /** @brief Returns the canonical parameter-vector length. @return Total weights and enabled biases. */
  std::size_t parameterCount() const;

  /**
   * @brief Applies a complete canonical parameter vector to all layers.
   * @param[in] parameters Layer layouts concatenated in propagation order.
   * @throws std::invalid_argument If the vector length differs from parameterCount().
   */
  void setParameters( Eigen::Ref< const Eigen::VectorXd > parameters );

  /** @brief Serializes all layer parameters. @return Weights and biases in canonical order. */
  Eigen::VectorXd parameters() const;
  /** @brief Serializes only the weights. @return Column-major weights concatenated in layer order. */
  Eigen::VectorXd weightsParameters() const;
  /** @brief Serializes only enabled biases. @return Biases concatenated in layer order. */
  Eigen::VectorXd biasesParameters() const;

  private:
  /** Validated copy used to construct the layer stack. */
  FeedForwardNetworkConfiguration configuration_;
  /** Dense layers in input-to-output propagation order. */
  std::vector< DenseLayer > layers_;
  /** Offsets and counts for the canonical serialized representation. */
  ParameterLayout parameter_layout_;
};


/** @} */

#endif // !FEED_FORWARD_NETWORK_H
