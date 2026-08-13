#ifndef OPTIMIZER_GPS_POLL_STRATEGY_H
#define OPTIMIZER_GPS_POLL_STRATEGY_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include "optimizations_methods/gps/GPSConfiguration.hpp"

/** @brief Interface that selects columns from the complete GPS poll directions. */
class GPSPollStrategy
{
  public:
  /** @brief Enables destruction through the strategy interface. */
  virtual ~GPSPollStrategy() = default;

  /**
   * @brief Selects poll directions for one GPS iteration.
   * @param[in] complete_directions Complete \f$D=GZ\f$ matrix, one direction per column.
   * @param[in] iteration Zero-based completed GPS iteration count.
   * @return Direction matrix whose columns are evaluated in order.
   */
  virtual Eigen::MatrixXd directions( Eigen::Ref< const Eigen::MatrixXd > complete_directions,
                                      std::size_t iteration ) const = 0;

  /** @brief Restores strategy state before a new run. */
  virtual void reset() = 0;
};

/** @brief Returns every complete positive-basis direction in original column order. */
class CompleteGPSPollStrategy : public GPSPollStrategy
{
  public:
  /** @brief Accepts the marker configuration. @param[in] configuration Complete-poll selection marker. */
  explicit CompleteGPSPollStrategy( CompleteGPSPollConfiguration configuration )
  {
    static_cast< void >( configuration );
  }

  /** @copydoc GPSPollStrategy::directions() */
  Eigen::MatrixXd directions( Eigen::Ref< const Eigen::MatrixXd > complete_directions,
                              std::size_t iteration ) const override
  {
    static_cast< void >( iteration );
    return complete_directions;
  }

  /** @brief Performs no work because the strategy is stateless. */
  void reset() override {}
};

/** @} */

#endif // !OPTIMIZER_GPS_POLL_STRATEGY_H
