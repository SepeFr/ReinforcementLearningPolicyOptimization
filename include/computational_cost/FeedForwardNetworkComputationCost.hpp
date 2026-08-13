#ifndef FEED_FORWARD_NETWORK_COMPUTATION_COST_H
#define FEED_FORWARD_NETWORK_COMPUTATION_COST_H

/** @addtogroup computational_cost_api
 * @{ */

#include <cstdint>
#include "FeedForwardNetworkConfiguration.hpp"

/**
 * @brief Counts feed-forward matrix multiply-accumulate operations.
 *
 * The count is the sum \f$\sum_{\ell=1}^{L}n_{\ell-1}n_\ell\f$ over hidden and output
 * layers. Bias additions and activation-function operations are outside this
 * metric.
 */
class FeedForwardNetworkComputationCost
{
  public:
  /**
   * @brief Returns the matrix multiply-accumulate count for one inference.
   * @param[in] configuration Layer widths used to form adjacent-layer products.
   * @return MAC operations for one complete feed-forward evaluation.
   * @throws std::overflow_error If a layer size cannot be represented as
   *         `std::uint64_t` or if a product or accumulated sum overflows.
   */
  static std::uint64_t macsPerInference( const FeedForwardNetworkConfiguration &configuration );

  private:
  /** @brief Prevents construction of this stateless utility class. */
  FeedForwardNetworkComputationCost() = delete;
};

/** @} */

#endif // !FEED_FORWARD_NETWORK_COMPUTATION_COST_H
