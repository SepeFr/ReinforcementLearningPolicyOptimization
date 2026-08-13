#ifndef EXPS_ENVIRONMENTS_PENDULUM_STOCHASTIC_HPP
#define EXPS_ENVIRONMENTS_PENDULUM_STOCHASTIC_HPP

#include <cstdint>
#include <random>
#include "Pendulum.hpp"

namespace exps::pendulum
{
  class StochasticPendulumEnvironment final : public PendulumEnvironment
  {
    public:
    StochasticPendulumEnvironment( std::uint64_t random_seed, double torque_noise_stddev );

    protected:
    double perturbTorque( double commanded_torque ) override;

    private:
    std::mt19937_64 noise_generator_;
    std::normal_distribution< double > torque_noise_distribution_;
  };
}

#endif
