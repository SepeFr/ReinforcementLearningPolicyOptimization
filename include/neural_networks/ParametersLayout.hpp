#ifndef PARAMETERS_LAYOUT_H
#define PARAMETERS_LAYOUT_H

/** @addtogroup neural_network_api
 * @{ */

#include <cstddef>
#include <iterator>
#include <vector>
/** @brief Offsets and counts for one layer in a network parameter vector. */
struct LayerParameterLayout
{
  /** @brief Constructs an empty layout whose offsets and counts are zero. */
  LayerParameterLayout() = default;

  std::size_t weights_offset = 0; ///< Index of the layer's first weight in the network vector.
  std::size_t weights_count = 0;  ///< Number of column-major weight entries.

  std::size_t biases_offset = 0; ///< Index of the layer's first bias in the network vector.
  std::size_t biases_count = 0;  ///< Number of bias entries; zero when bias is disabled.
};

/**
 * @brief Records the per-layer slices of a serialized network parameter vector.
 *
 * Layer indices follow forward-propagation order. Within each layer, weights
 * precede biases.
 */
class ParameterLayout
{
  public:
  /** @brief Constructs a layout with no layers. */
  ParameterLayout() = default;
  /**
   * @brief Constructs zero-initialized entries for a fixed layer count.
   * @param[in] number_of_layers Number of addressable layer entries.
   */
  ParameterLayout( std::size_t number_of_layers );
  /** @brief Sums all recorded weight and bias counts. @return Serialized network length. */
  std::size_t totalParameterCount() const;

  /**
   * @brief Returns one layer entry.
   * @param[in] index Zero-based layer index in propagation order.
   * @return Read-only entry owned by this layout.
   * @throws std::out_of_range If @p index is outside the layout.
   */
  const LayerParameterLayout &layer( std::size_t index ) const;
  /**
   * @brief Replaces one layer entry.
   * @param[in] index Zero-based layer index in propagation order.
   * @param[in] layout Offsets and counts to store by value.
   * @throws std::out_of_range If @p index is outside the layout.
   */
  void setLayer( std::size_t index, const LayerParameterLayout &layout );

  private:
  /** Per-layer entries in propagation order. */
  std::vector< LayerParameterLayout > layout_parameters_;
};

/** @} */

#endif // !PARAMETERS_LAYOUT_H
