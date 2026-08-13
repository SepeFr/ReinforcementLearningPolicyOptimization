#ifndef OPTIMIZER_STOCHASTIC_GPS_STOPPING_CRITERION_H
#define OPTIMIZER_STOCHASTIC_GPS_STOPPING_CRITERION_H

/** @addtogroup optimization_methods_api
 * @{ */

/** @file StochasticGPSStoppingCriterion.hpp @brief Stochastic GPS stopping state and strategy alternatives. */

#include <cmath>
#include <optional>
#include <variant>

/** @brief Disables the stochastic GPS internal stopping test. */
struct NoStochasticGPSStoppingConfiguration
{
};

/** @brief Uses GeneralizedPatternSearchHyperparameters::stopping_mesh_size as an inclusive threshold. */
struct StochasticGPSMeshSizeStoppingConfiguration
{
};

/** @brief Stops when decayed ranking-and-selection significance reaches a tolerance. */
struct StochasticGPSSignificanceStoppingConfiguration
{
  double tolerance = 1.0e-6; ///< Finite threshold strictly between zero and one; comparison is inclusive.
};

/** @brief Stops when incumbent noise dominates the current indifference width. */
struct StochasticGPSNoiseToIndifferenceStoppingConfiguration
{
  double threshold = 1.0e6; ///< Finite positive \f$L\f$ in \f$S_k/\delta_r\geq\sqrt L\f$.
};

/** @brief Variant selecting the internal convergence test for stochastic GPS. */
using StochasticGPSStoppingConfiguration =
  std::variant< NoStochasticGPSStoppingConfiguration,
                StochasticGPSMeshSizeStoppingConfiguration,
                StochasticGPSSignificanceStoppingConfiguration,
                StochasticGPSNoiseToIndifferenceStoppingConfiguration >;

/** @brief Snapshot consumed by a stochastic GPS stopping strategy. */
struct StochasticGPSStoppingState
{
  double mesh_size; ///< Current GPS mesh size \f$\delta_k\f$.
  double significance; ///< Current ranking-and-selection significance \f$\alpha_r\f$.
  double indifference; ///< Current positive indifference width \f$\delta_r\f$.
  std::optional< double > incumbent_standard_deviation; ///< Bessel-corrected \f$S_k\f$ when available.
};

/** @brief Interface for an internal stochastic GPS convergence test. */
class StochasticGPSStoppingStrategy
{
  public:
  /** @brief Enables destruction through the strategy interface. */
  virtual ~StochasticGPSStoppingStrategy() = default;

  /** @brief Reports whether this strategy defines a criterion. @return `true` by default. */
  virtual bool hasCriterion() const { return true; }
  /**
   * @brief Tests a current algorithm snapshot.
   * @param[in] state Mesh, statistical schedule, and optional noise estimate.
   * @return Whether the run has converged.
   */
  virtual bool shouldStop( const StochasticGPSStoppingState &state ) const = 0;
};

/** @brief Marker strategy that always remains active without an internal criterion. */
class NoStochasticGPSStoppingCriterion final
  : public StochasticGPSStoppingStrategy
{
  public:
  /** @brief Reports absence of an internal criterion. @return `false`. */
  bool hasCriterion() const override { return false; }
  /** @brief Never stops. @return `false`. */
  bool shouldStop( const StochasticGPSStoppingState & ) const override
  {
    return false;
  }
};

/** @brief Stops when mesh size is at most a stored tolerance. */
class StochasticGPSMeshSizeStoppingCriterion final
  : public StochasticGPSStoppingStrategy
{
  public:
  /** @brief Stores the mesh threshold. @param[in] tolerance Inclusive positive mesh limit. */
  explicit StochasticGPSMeshSizeStoppingCriterion( double tolerance ) :
      tolerance_( tolerance )
  {
  }

  /**
   * @brief Tests the mesh-size component.
   * @param[in] state Current stochastic GPS state.
   * @return Whether `state.mesh_size <= tolerance_`.
   */
  bool shouldStop( const StochasticGPSStoppingState &state ) const override
  {
    return state.mesh_size <= tolerance_;
  }

  private:
  /** Inclusive mesh-size threshold. */
  double tolerance_;
};

/** @brief Stops when current significance is at most a configured tolerance. */
class StochasticGPSSignificanceStoppingCriterion final
  : public StochasticGPSStoppingStrategy
{
  public:
  /** @brief Stores the significance threshold. @param[in] configuration Validated threshold. */
  explicit StochasticGPSSignificanceStoppingCriterion(
    StochasticGPSSignificanceStoppingConfiguration configuration ) :
      configuration_( configuration )
  {
  }

  /**
   * @brief Tests the significance component.
   * @param[in] state Current stochastic GPS state.
   * @return Whether significance is at most the configured tolerance.
   */
  bool shouldStop( const StochasticGPSStoppingState &state ) const override
  {
    return state.significance <= configuration_.tolerance;
  }

  private:
  /** Copied significance threshold. */
  StochasticGPSSignificanceStoppingConfiguration configuration_;
};

/** @brief Stops when \f$S_k/\delta_r\f$ reaches \f$\sqrt L\f$. */
class StochasticGPSNoiseToIndifferenceStoppingCriterion final
  : public StochasticGPSStoppingStrategy
{
  public:
  /** @brief Stores the ratio threshold. @param[in] configuration Validated positive threshold. */
  explicit StochasticGPSNoiseToIndifferenceStoppingCriterion(
    StochasticGPSNoiseToIndifferenceStoppingConfiguration configuration ) :
      configuration_( configuration )
  {
  }

  /**
   * @brief Tests incumbent noise relative to the indifference width.
   * @param[in] state Current stochastic GPS state.
   * @return `false` without a standard deviation; otherwise the inclusive ratio test.
   */
  bool shouldStop( const StochasticGPSStoppingState &state ) const override
  {
    if ( !state.incumbent_standard_deviation.has_value() )
    {
      return false;
    }

    const double noise_to_indifference =
      *state.incumbent_standard_deviation / state.indifference;
    return noise_to_indifference >= std::sqrt( configuration_.threshold );
  }

  private:
  /** Copied squared ratio threshold. */
  StochasticGPSNoiseToIndifferenceStoppingConfiguration configuration_;
};

/** @} */

#endif // !OPTIMIZER_STOCHASTIC_GPS_STOPPING_CRITERION_H
