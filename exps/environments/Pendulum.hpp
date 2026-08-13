#ifndef EXPS_ENVIRONMENTS_PENDULUM_HPP
#define EXPS_ENVIRONMENTS_PENDULUM_HPP

#include <cstddef>
#include <eigen3/Eigen/Core>
#include "simulator/ActionBase.hpp"
#include "simulator/Environment.hpp"
#include "simulator/ObservationBase.hpp"
#include "simulator/TerminationReason.hpp"

namespace exps::pendulum
{
  struct Scenario
  {
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
    double torque() const;
  };

  class PendulumEnvironment : public Environment< Scenario, Observation, Action, ::TerminationReason >
  {
    public:
    using Base = Environment< Scenario, Observation, Action, ::TerminationReason >;
    using Step = typename Base::StepResult;

    Observation reset( const Scenario &scenario ) override;
    Step step( const Action &action ) override;

    protected:
    virtual double perturbTorque( double commanded_torque );

    private:
    static double angleNormalize( double angle );

    inline static constexpr double gravity_ = 10.0;
    inline static constexpr double mass_ = 1.0;
    inline static constexpr double length_ = 1.0;
    inline static constexpr double time_step_ = 0.05;
    inline static constexpr double maximum_torque_ = 2.0;
    inline static constexpr double maximum_speed_ = 8.0;
    inline static constexpr std::size_t maximum_steps_ = 200;

    Scenario state_;
    std::size_t step_count_ = 0;
  };
}

#endif
