#include "PendulumStochastic.hpp"

#include <cmath>
#include <stdexcept>

namespace
{
  double requirePositiveNoiseStddev( double value )
  {
    if ( !std::isfinite( value ) || value <= 0.0 )
    {
      throw std::invalid_argument( "Stochastic Pendulum torque noise must be finite and positive" );
    }
    return value;
  }
}

namespace exps::pendulum
{
  StochasticPendulumEnvironment::StochasticPendulumEnvironment(
    std::uint64_t random_seed, double torque_noise_stddev ) :
      noise_generator_( random_seed ),
      torque_noise_distribution_( 0.0, requirePositiveNoiseStddev( torque_noise_stddev ) )
  {}

  double StochasticPendulumEnvironment::perturbTorque( double commanded_torque )
  {
    return commanded_torque + torque_noise_distribution_( noise_generator_ );
  }
}
