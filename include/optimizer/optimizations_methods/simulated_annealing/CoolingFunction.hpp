#ifndef OPTIMIZER_SIMULATED_ANNEALING_COOLING_FUNCTION_H
#define OPTIMIZER_SIMULATED_ANNEALING_COOLING_FUNCTION_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "optimizations_methods/simulated_annealing/CoolingFunctionConfiguration.hpp"

/** @brief Stateful interface for a simulated-annealing temperature schedule. */
class CoolingFunction
{
  public:
  /**
   * @brief Stores the initial and last-reported temperature.
   * @param[in] initial_temperature Schedule scale supplied by the method.
   */
  explicit CoolingFunction( double initial_temperature ) :
      initial_temperature_( initial_temperature ), last_temperature_( initial_temperature )
  {
  }

  /** @brief Enables destruction through the schedule interface. */
  virtual ~CoolingFunction() = default;

  /**
   * @brief Evaluates the schedule at an algorithmic time.
   * @param[in] iteration Time index supplied by SimulatedAnnealingMethod.
   * @return Computed temperature; also stored in last_temperature_.
   */
  virtual double temperature( std::size_t iteration ) = 0;

  protected:
  /** Construction temperature \f$T_0\f$. */
  double initial_temperature_;
  /** Most recently computed temperature. */
  double last_temperature_;
};

/** @brief Computes \f$T_k=T_0\alpha^k\f$. */
class GeometricCooling : public CoolingFunction
{
  public:
  /** @brief Stores schedule parameters. @param[in] initial_temperature Initial \f$T_0\f$. @param[in] configuration Geometric decay factor. */
  GeometricCooling( double initial_temperature, GeometricCoolingConfiguration configuration ) :
      CoolingFunction( initial_temperature ), configuration_( configuration )
  {
  }

  /** @copydoc CoolingFunction::temperature() */
  double temperature( std::size_t iteration ) override
  {
    last_temperature_ = initial_temperature_ * std::pow( configuration_.alpha, iteration );
    return last_temperature_;
  }

  private:
  /** Geometric decay factor. */
  GeometricCoolingConfiguration configuration_;
};

/** @brief Computes \f$T_k=\max\{T_{\min},T_0-\beta k\}\f$. */
class LinearCooling : public CoolingFunction
{
  public:
  /** @brief Stores schedule parameters. @param[in] initial_temperature Initial \f$T_0\f$. @param[in] configuration Slope and floor. */
  LinearCooling( double initial_temperature, LinearCoolingConfiguration configuration ) :
      CoolingFunction( initial_temperature ), configuration_( configuration )
  {
  }

  /** @copydoc CoolingFunction::temperature() */
  double temperature( std::size_t iteration ) override
  {
    last_temperature_ = std::max( configuration_.minimum_temperature,
                                  initial_temperature_ - static_cast< double >( iteration ) * configuration_.beta );
    return last_temperature_;
  }

  private:
  /** Linear slope and temperature floor. */
  LinearCoolingConfiguration configuration_;
};

/** @brief Computes \f$T_k=c/\log(k+d)\f$. */
class LogarithmicCooling : public CoolingFunction
{
  public:
  /** @brief Stores schedule parameters. @param[in] initial_temperature Retained base value. @param[in] configuration Numerator and time offset. */
  LogarithmicCooling( double initial_temperature, LogarithmicCoolingConfiguration configuration ) :
      CoolingFunction( initial_temperature ), configuration_( configuration )
  {
  }

  /** @copydoc CoolingFunction::temperature() */
  double temperature( std::size_t iteration ) override
  {
    last_temperature_ = configuration_.c / std::log( static_cast< double >( iteration ) + configuration_.d );
    return last_temperature_;
  }

  private:
  /** Logarithmic numerator and time offset. */
  LogarithmicCoolingConfiguration configuration_;
};

/** @brief Computes \f$T_k=T_0/(1+bk)\f$. */
class ReciprocalCooling : public CoolingFunction
{
  public:
  /** @brief Stores schedule parameters. @param[in] initial_temperature Initial \f$T_0\f$. @param[in] configuration Denominator growth. */
  ReciprocalCooling( double initial_temperature, ReciprocalCoolingConfiguration configuration ) :
      CoolingFunction( initial_temperature ), configuration_( configuration )
  {
  }

  /** @copydoc CoolingFunction::temperature() */
  double temperature( std::size_t iteration ) override
  {
    last_temperature_ = initial_temperature_ / ( 1.0 + configuration_.b * static_cast< double >( iteration ) );
    return last_temperature_;
  }

  private:
  /** Reciprocal denominator coefficient. */
  ReciprocalCoolingConfiguration configuration_;
};

/** @brief Computes \f$T_k=T_0\exp(-\beta k)\f$. */
class ExponentialCooling : public CoolingFunction
{
  public:
  /** @brief Stores schedule parameters. @param[in] initial_temperature Initial \f$T_0\f$. @param[in] configuration Exponential rate. */
  ExponentialCooling( double initial_temperature, ExponentialCoolingConfiguration configuration ) :
      CoolingFunction( initial_temperature ), configuration_( configuration )
  {
  }

  /** @copydoc CoolingFunction::temperature() */
  double temperature( std::size_t iteration ) override
  {
    last_temperature_ = initial_temperature_ * std::exp( -configuration_.beta * static_cast< double >( iteration ) );
    return last_temperature_;
  }

  private:
  /** Exponential decay rate. */
  ExponentialCoolingConfiguration configuration_;
};

/** @brief Computes \f$T_k=(n-1)/\log(k+1)\f$ for configured dimension \f$n\f$. */
class InverseLogarithmicCooling : public CoolingFunction
{
  public:
  /** @brief Stores schedule parameters. @param[in] initial_temperature Retained base value. @param[in] configuration Effective dimension. */
  InverseLogarithmicCooling( double initial_temperature, InverseLogarithmicCoolingConfiguration configuration ) :
      CoolingFunction( initial_temperature ), configuration_( configuration )
  {
  }

  /** @copydoc CoolingFunction::temperature() */
  double temperature( std::size_t iteration ) override
  {
    last_temperature_ =
      ( configuration_.dimension - 1.0 ) / std::log( static_cast< double >( iteration ) + 1.0 );
    return last_temperature_;
  }

  private:
  /** Effective dimension controlling the numerator. */
  InverseLogarithmicCoolingConfiguration configuration_;
};

/** @} */

#endif // !OPTIMIZER_SIMULATED_ANNEALING_COOLING_FUNCTION_H
