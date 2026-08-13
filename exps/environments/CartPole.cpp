#include "CartPole.hpp"

#include <cmath>
#include <stdexcept>

namespace exps::cartpole
{
  Observation::Observation( const Scenario &state )
  {
    value().resize( 4 );
    value() << state.x, state.x_velocity, state.theta, state.theta_velocity;
  }

  void Action::vectorToAction( Eigen::Ref< const Eigen::VectorXd > vector )
  {
    if ( vector.size() != 1 )
    {
      throw std::invalid_argument( "CartPole action requires exactly one policy output" );
    }

    value().resize( 1 );
    value()[0] = vector[0] < 0.0 ? -10.0 : 10.0;
  }

  double Action::force() const
  {
    if ( value().size() != 1 )
    {
      throw std::logic_error( "CartPole action has not been initialized" );
    }
    return value()[0];
  }

  Observation CartPoleEnvironment::reset( const Scenario &scenario )
  {
    state_ = scenario;
    step_count_ = 0;
    return Observation( state_ );
  }

  CartPoleEnvironment::Step CartPoleEnvironment::step( const Action &action )
  {
    const double force = action.force();
    const double sine = std::sin( state_.theta );
    const double cosine = std::cos( state_.theta );
    const double temporary =
      ( force + pole_mass_length_ * state_.theta_velocity * state_.theta_velocity * sine ) / total_mass_;
    const double theta_acceleration =
      ( gravity_ * sine - cosine * temporary ) /
      ( half_pole_length_ * ( 4.0 / 3.0 - pole_mass_ * cosine * cosine / total_mass_ ) );
    const double x_acceleration =
      temporary - pole_mass_length_ * theta_acceleration * cosine / total_mass_;

    const Scenario next_state{
      state_.x + time_step_ * state_.x_velocity,
      state_.x_velocity + time_step_ * x_acceleration,
      state_.theta + time_step_ * state_.theta_velocity,
      state_.theta_velocity + time_step_ * theta_acceleration
    };

    state_ = next_state;
    ++step_count_;

    const bool terminated =
      std::abs( state_.x ) > position_threshold_ || std::abs( state_.theta ) > angle_threshold_radians_;
    const bool truncated = step_count_ >= maximum_steps_;

    Step result;
    result.observation = Observation( state_ );
    result.reward = 1.0;
    result.terminated = terminated;
    result.truncated = truncated;
    if ( terminated )
    {
      result.termination_reason = ::TerminationReason::Failure;
    }
    else if ( truncated )
    {
      result.termination_reason = ::TerminationReason::TimeLimitReached;
    }
    return result;
  }
}
