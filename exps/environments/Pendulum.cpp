#include "Pendulum.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace exps::pendulum
{
  Observation::Observation( const Scenario &state )
  {
    value().resize( 3 );
    value() << std::cos( state.theta ), std::sin( state.theta ), state.theta_velocity;
  }

  void Action::vectorToAction( Eigen::Ref< const Eigen::VectorXd > vector )
  {
    if ( vector.size() != 1 )
    {
      throw std::invalid_argument( "Pendulum action requires exactly one policy output" );
    }

    value().resize( 1 );
    value()[0] = std::clamp( 2.0 * vector[0], -2.0, 2.0 );
  }

  double Action::torque() const
  {
    if ( value().size() != 1 )
    {
      throw std::logic_error( "Pendulum action has not been initialized" );
    }
    return value()[0];
  }

  Observation PendulumEnvironment::reset( const Scenario &scenario )
  {
    state_ = scenario;
    step_count_ = 0;
    return Observation( state_ );
  }

  PendulumEnvironment::Step PendulumEnvironment::step( const Action &action )
  {
    const double commanded_torque = std::clamp( action.torque(), -maximum_torque_, maximum_torque_ );
    const double applied_torque =
      std::clamp( perturbTorque( commanded_torque ), -maximum_torque_, maximum_torque_ );
    const double normalized_angle = angleNormalize( state_.theta );
    const double cost = normalized_angle * normalized_angle +
      0.1 * state_.theta_velocity * state_.theta_velocity +
      0.001 * commanded_torque * commanded_torque;

    const double raw_next_theta_velocity = state_.theta_velocity +
      ( ( 3.0 * gravity_ / ( 2.0 * length_ ) ) * std::sin( state_.theta ) +
        ( 3.0 / ( mass_ * length_ * length_ ) ) * applied_torque ) *
        time_step_;
    const double next_theta_velocity =
      std::clamp( raw_next_theta_velocity, -maximum_speed_, maximum_speed_ );
    const double next_theta = state_.theta + next_theta_velocity * time_step_;

    state_ = Scenario{ next_theta, next_theta_velocity };
    ++step_count_;

    Step result;
    result.observation = Observation( state_ );
    result.reward = -cost;
    result.terminated = false;
    result.truncated = step_count_ >= maximum_steps_;
    if ( result.truncated )
    {
      result.termination_reason = ::TerminationReason::TimeLimitReached;
    }
    return result;
  }

  double PendulumEnvironment::perturbTorque( double commanded_torque )
  {
    return commanded_torque;
  }

  double PendulumEnvironment::angleNormalize( double angle )
  {
    constexpr double pi = std::numbers::pi_v< double >;
    constexpr double two_pi = 2.0 * pi;
    double shifted_remainder = std::fmod( angle + pi, two_pi );
    if ( shifted_remainder < 0.0 )
    {
      shifted_remainder += two_pi;
    }
    return shifted_remainder - pi;
  }
}
