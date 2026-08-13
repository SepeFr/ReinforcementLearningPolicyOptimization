#ifndef EXPS_ENVIRONMENTS_CARTPOLE_HPP
#define EXPS_ENVIRONMENTS_CARTPOLE_HPP

#include <cstddef>
#include <eigen3/Eigen/Core>
#include "simulator/ActionBase.hpp"
#include "simulator/Environment.hpp"
#include "simulator/ObservationBase.hpp"
#include "simulator/TerminationReason.hpp"

namespace exps::cartpole
{
  struct Scenario
  {
    double x = 0.0;
    double x_velocity = 0.0;
    double theta = 0.0;
    double theta_velocity = 0.0;
  };

  class Observation final : public ObservationBase
  {
    public:
    Observation() = default;
    explicit Observation( const Scenario &state );
  };

  class Action final : public ActionBase
  {
    public:
    Action() = default;

    void vectorToAction( Eigen::Ref< const Eigen::VectorXd > vector ) override;
    double force() const;
  };

  class CartPoleEnvironment final : public Environment< Scenario, Observation, Action, ::TerminationReason >
  {
    public:
    using Base = Environment< Scenario, Observation, Action, ::TerminationReason >;
    using Step = typename Base::StepResult;

    Observation reset( const Scenario &scenario ) override;
    Step step( const Action &action ) override;

    private:
    inline static constexpr double gravity_ = 9.8;
    inline static constexpr double cart_mass_ = 1.0;
    inline static constexpr double pole_mass_ = 0.1;
    inline static constexpr double half_pole_length_ = 0.5;
    inline static constexpr double total_mass_ = cart_mass_ + pole_mass_;
    inline static constexpr double pole_mass_length_ = pole_mass_ * half_pole_length_;
    inline static constexpr double time_step_ = 0.02;
    inline static constexpr double position_threshold_ = 2.4;
    inline static constexpr double angle_threshold_radians_ = 0.20943951023931953;
    inline static constexpr std::size_t maximum_steps_ = 500;

    Scenario state_;
    std::size_t step_count_ = 0;
  };
}

#endif
