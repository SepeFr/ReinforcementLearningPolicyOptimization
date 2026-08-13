#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

/** @addtogroup neural_network_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include "ActivationType.hpp"
/**
 * @brief Implements one dense affine layer followed by an activation.
 *
 * For input \f$h^{(\ell-1)}\f$, weights \f$W^{(\ell)}\f$, and optional bias
 * \f$b^{(\ell)}\f$, forward() computes
 * \f$\phi_\ell(W^{(\ell)}h^{(\ell-1)}+b^{(\ell)})\f$.
 * `W` has `output_size` rows and `input_size`
 * columns. Parameters are stored and exported as the column-major entries of
 * `W`, followed by `b` when bias is enabled.
 *
 * @see ParameterLayout
 * @see neural_parameter_mapping_chapter
 */
class DenseLayer
{
  public:
  /**
   * @brief Constructs a zero-initialized dense layer.
   * @param[in] input_size Number of input components.
   * @param[in] output_size Number of output components.
   * @param[in] activation_type Element-wise activation applied to the affine result.
   * @param[in] use_bias Whether to allocate and apply one bias per output.
   * @throws std::overflow_error If either dimension cannot be represented as Eigen::Index.
   */
  DenseLayer( std::size_t input_size, std::size_t output_size, ActivationType activation_type, bool use_bias );

  /**
   * @brief Propagates one vector through the layer.
   * @param[in] input Vector with input_size components.
   * @return Activated output with output_size components.
   * @throws std::invalid_argument If the input dimension differs from the weight-column count.
   */
  Eigen::VectorXd forward( Eigen::Ref< const Eigen::VectorXd > input ) const;

  /** @brief Returns the serialized weight-and-bias count. @return weightCount() plus biasCount(). */
  std::size_t parameterCount() const;
  /** @brief Returns the number of entries in the weight matrix. @return `input_size * output_size`. */
  std::size_t weightCount() const;
  /** @brief Returns the stored bias count. @return `output_size` when enabled, otherwise zero. */
  std::size_t biasCount() const;

  /**
   * @brief Replaces weights and biases from the canonical layer layout.
   * @param[in] parameters Column-major weights followed by biases.
   * @throws std::invalid_argument If the vector length differs from parameterCount().
   * @post A subsequent appendParameters() reproduces @p parameters.
   */
  void setParameters( Eigen::Ref< const Eigen::VectorXd > parameters );
  /**
   * @brief Appends the complete canonical layer layout to a vector.
   * @param[in,out] destination Existing entries are retained; weights and then biases are appended.
   */
  void appendParameters( Eigen::VectorXd &destination ) const;

  /**
   * @brief Appends column-major weights to a vector.
   * @param[in,out] destination Existing entries are retained.
   */
  void appendWeights(Eigen::VectorXd &destination) const;
  /**
   * @brief Appends biases to a vector when bias is enabled.
   * @param[in,out] destination Existing entries are retained.
   */
  void appendBiases(Eigen::VectorXd &destination) const;

  private:
  /** Weight matrix with one row per output and one column per input. */
  Eigen::MatrixXd weights_;
  /** Bias vector, empty when bias use is disabled. */
  Eigen::VectorXd biases_;

  /** Element-wise activation selected at construction. */
  ActivationType activation_;
  /** Controls bias storage and addition. */
  bool use_bias_;
};

/** @} */

#endif // !DENSE_LAYER_H
