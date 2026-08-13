#ifndef FEED_FORWARD_POLICY_H
#define FEED_FORWARD_POLICY_H

/** @addtogroup neural_network_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <span>
#include <stdexcept>
#include <type_traits>
#include "ActionBase.hpp"
#include "FeedForwardNetwork.hpp"
#include "FeedForwardNetworkConfiguration.hpp"
#include "ObservationBase.hpp"
#include "ParametrizedPolicy.hpp"

/**
 * @brief Adapts a FeedForwardNetwork to the typed Policy interface.
 *
 * Observations are encoded with ObservationBase::observationToVector(). The
 * network output is copied into a default-constructed action through
 * ActionBase::vectorToAction().
 *
 * @tparam ObservationType Observation derived from ObservationBase.
 * @tparam ActionType Default-constructible action derived from ActionBase.
 * @see neural_parameter_mapping_chapter
 */
template< typename ObservationType = ObservationBase, typename ActionType = ActionBase >
class FeedForwardPolicy : public ParametrizedPolicy< ObservationType, ActionType >
{
  static_assert( std::is_base_of_v< ObservationBase, ObservationType >,
                 "FeedForwardPolicy: ObservationType must derive from ObservationBase" );
  static_assert( std::is_base_of_v< ActionBase, ActionType >,
                 "FeedForwardPolicy: ActionType must derive from ActionBase" );
  static_assert( std::is_default_constructible_v< ActionType >,
                 "FeedForwardPolicy: ActionType must be default constructible" );

  public:
  using Observation = ObservationType; ///< Observation type accepted by act().
  using Action = ActionType;           ///< Action type returned by act().

  /**
   * @brief Constructs the owned network from a configuration.
   * @param[in] configuration Network structure copied into the policy.
   * @throws std::invalid_argument If a configured layer width is zero.
   * @throws std::overflow_error If dimensions or counts exceed supported ranges.
   */
  explicit FeedForwardPolicy( const FeedForwardNetworkConfiguration &configuration ) : network_( configuration ) {}

  /** @copydoc ParametrizedPolicy::parameterCount() */
  std::size_t parameterCount() const override { return network_.parameterCount(); }

  /**
   * @brief Applies a contiguous canonical network parameter sequence.
   * @param[in] parameters Exactly parameterCount() values.
   * @throws std::invalid_argument If the span length differs from parameterCount().
   */
  void setParameters( std::span< const double > parameters ) override
  {
    const Eigen::Map< const Eigen::VectorXd > parameter_map( parameters.data(),
                                                             static_cast< Eigen::Index >( parameters.size() ) );
    network_.setParameters( parameter_map );
  }

  /**
   * @brief Applies an Eigen canonical network parameter vector.
   * @param[in] parameters Exactly parameterCount() values.
   * @throws std::invalid_argument If the vector length differs from parameterCount().
   */
  void setParameters( Eigen::Ref< const Eigen::VectorXd > parameters ) override
  {
    network_.setParameters( parameters );
  }

  /** @copydoc ParametrizedPolicy::parameters() */
  Eigen::VectorXd parameters() const override { return network_.parameters(); }

  /**
   * @brief Computes an action from one observation.
   * @param[in] observation Observation encoded in the configured input order.
   * @return Action populated from the network output.
   * @throws std::invalid_argument If the encoded observation has the wrong dimension.
   * @throws std::overflow_error If any network output component is non-finite.
   */
  Action act( const Observation &observation ) override
  {
    const Eigen::VectorXd input = observation.observationToVector();
    const Eigen::VectorXd output = network_.forward( input );
    if ( !output.allFinite() )
    {
      throw std::overflow_error( "FeedForwardPolicy: network output must be finite" );
    }

    Action action;
    action.vectorToAction( output );
    return action;
  }

  private:
  /** Network owned by and sharing the lifetime of this policy. */
  FeedForwardNetwork network_;
};

/** @} */

#endif // !FEED_FORWARD_POLICY_H
