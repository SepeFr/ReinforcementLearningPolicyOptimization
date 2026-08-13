#ifndef OPTIMIZER_NELDER_MEAD_STOPPING_CRITERION_H
#define OPTIMIZER_NELDER_MEAD_STOPPING_CRITERION_H

/** @addtogroup optimization_methods_api
 * @{ */

/** @file NelderMeadStoppingCriterion.hpp @brief Simplex convergence configuration and strategy alternatives. */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

/** @brief Configures simultaneous coordinate and objective spread tolerances. */
struct DispersionStoppingConfiguration
{
  double x_absolute_tolerance = 1.0e-4; ///< Finite non-negative maximum coordinate spread.
  double f_absolute_tolerance = 1.0e-4; ///< Finite non-negative maximum objective spread.
};

/** @brief Configures relative variation of consecutive best simplex values. */
struct RelativeVariationStoppingConfiguration
{
  double tolerance = 1.0e-8; ///< Finite non-negative strict relative-variation threshold.
};

/** @brief Configures a strict standard-deviation threshold for simplex values. */
struct StandardDeviationStoppingConfiguration
{
  double tolerance = 1.0e-8; ///< Finite non-negative threshold in objective units.
};

/** @brief Configures simplex height and optional absolute-best termination tests. */
struct HeightDifferenceStoppingConfiguration
{
  double convergence_tolerance = 1.0e-8; ///< Finite non-negative allowed worst-minus-best height.

  /**
   * Stops when the best value of the internal minimization objective
   * is less than or equal to this threshold.
   *
   * For maximization problems, the internal objective is -f(x);
   * therefore, a target f(x) >= T must be specified as -T.
   *
   * Negative infinity disables this condition.
   */
  double absolute_tolerance = -std::numeric_limits< double >::infinity(); ///< Internal minimization target; negative infinity disables it.
};

/** @brief Configures scale-robust change of consecutive best values. */
struct RelativeFunctionToleranceStoppingConfiguration
{
  double relative_tolerance = 1.0e-8; ///< Finite non-negative strict relative-change threshold.
};

/** @brief Variant selecting one simplex-owned convergence test. */
using NelderMeadStoppingConfiguration =
  std::variant< DispersionStoppingConfiguration, RelativeVariationStoppingConfiguration,
                StandardDeviationStoppingConfiguration, HeightDifferenceStoppingConfiguration,
                RelativeFunctionToleranceStoppingConfiguration >;

/** @brief Interface for convergence tests on a complete ordered simplex. */
class NelderMeadStoppingStrategy
{
  public:
  /** @brief Enables destruction through the strategy interface. */
  virtual ~NelderMeadStoppingStrategy() = default;

  /**
   * @brief Tests a simplex ordered from lowest to highest minimization mean.
   * @param[in] ordered_simplex Vertex parameters in the same order as @p ordered_values.
   * @param[in] ordered_values Successful vertex means from best to worst.
   * @return `true` when the configured convergence condition is satisfied.
   * @pre Both vectors have equal nonzero cardinality and matching indices.
   */
  virtual bool shouldStop( const std::vector< Eigen::VectorXd > &ordered_simplex,
                           const std::vector< double > &ordered_values ) = 0;
};

/*
 * Inspired by the SciPy Nelder-Mead convergence criterion:
 *
 * max_{i=1,...,n; j=1,...,n} |x_{i,j} - x_{0,j}| <= xatol
 *
 * and
 *
 * max_{i=1,...,n} |f(x_i) - f(x_0)| <= fatol
 */
/**
 * @brief Requires both coordinate and value dispersion to meet absolute limits.
 *
 * With best vertex \f$x_0\f$, the test is
 * \f$\max_{i,j}|x_{i,j}-x_{0,j}|\leq x_{tol}\f$ and
 * \f$\max_i|f_i-f_0|\leq f_{tol}\f$.
 */
class DispersionStoppingCriterion final : public NelderMeadStoppingStrategy
{
  public:
  /** @brief Stores dispersion tolerances. @param[in] configuration Coordinate and objective limits. */
  explicit DispersionStoppingCriterion( DispersionStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /** @copydoc NelderMeadStoppingStrategy::shouldStop() */
  bool shouldStop( const std::vector< Eigen::VectorXd > &ordered_simplex,
                   const std::vector< double > &ordered_values ) override
  {
    double maximum_x_difference = 0.0;
    double maximum_f_difference = 0.0;

    for ( std::size_t vertex_index = 1; vertex_index < ordered_simplex.size(); vertex_index++ )
    {
      maximum_x_difference = std::max(
        maximum_x_difference, ( ordered_simplex.at( vertex_index ) - ordered_simplex.front() ).cwiseAbs().maxCoeff() );
      maximum_f_difference =
        std::max( maximum_f_difference, std::abs( ordered_values.at( vertex_index ) - ordered_values.front() ) );
    }

    return maximum_x_difference <= configuration_.x_absolute_tolerance &&
      maximum_f_difference <= configuration_.f_absolute_tolerance;
  }

  private:
  /** Copied absolute dispersion limits. */
  DispersionStoppingConfiguration configuration_;
};

/*
 * Relative variation between the best values of two consecutive simplexes:
 *
 * |f_{k+1} - f_k| / ((|f_{k+1}| + |f_k|) / 2) < tolerance
 *
 * Source: Numerical Recipes fractional-change stopping test.
 */
/**
 * @brief Tests the symmetric relative change of consecutive best values.
 *
 * After the first observation, the criterion computes
 * \f$|f_k-f_{k-1}|/((|f_k|+|f_{k-1}|)/2)\f$. A zero denominator produces
 * zero variation. The comparison with tolerance is strict.
 */
class RelativeVariationStoppingCriterion final : public NelderMeadStoppingStrategy
{
  public:
  /** @brief Stores the relative-variation tolerance. @param[in] configuration Validated tolerance. */
  explicit RelativeVariationStoppingCriterion( RelativeVariationStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Updates the previous-best state and tests relative variation.
   * @param[in] ordered_values Values ordered from best to worst.
   * @return `false` on the first call; subsequently the strict tolerance result.
   */
  bool shouldStop( const std::vector< Eigen::VectorXd > &, const std::vector< double > &ordered_values ) override
  {
    const double current_best_value = ordered_values.front();
    if ( !previous_best_value_.has_value() )
    {
      previous_best_value_ = current_best_value;
      return false;
    }

    const double difference = std::abs( current_best_value - *previous_best_value_ );
    const double average_magnitude = ( std::abs( current_best_value ) + std::abs( *previous_best_value_ ) ) / 2.0;
    previous_best_value_ = current_best_value;

    const double relative_variation = average_magnitude == 0.0 ? 0.0 : difference / average_magnitude;
    return relative_variation < configuration_.tolerance;
  }

  private:
  /** Copied strict relative-variation threshold. */
  RelativeVariationStoppingConfiguration configuration_;
  /** Best mean from the preceding complete simplex. */
  std::optional< double > previous_best_value_;
};

/*
 * Standard deviation of the n+1 function values over the simplex:
 *
 * mean_f = (1 / (n + 1)) * sum_{i=0}^{n} f(x_i)
 * sqrt((1 / n) * sum_{i=0}^{n} (f(x_i) - mean_f)^2) < tolerance
 *
 * Source: Nelder-Mead standard-error criterion reported by J. C. Nash.
 */
/**
 * @brief Tests dispersion of all \f$n+1\f$ simplex objective values.
 *
 * The computed value is
 * \f$\sqrt{\sum_{i=0}^{n}(f_i-\bar f)^2/n}\f$ and is compared strictly with
 * the configured tolerance.
 */
class StandardDeviationStoppingCriterion final : public NelderMeadStoppingStrategy
{
  public:
  /** @brief Stores the objective-dispersion tolerance. @param[in] configuration Validated tolerance. */
  explicit StandardDeviationStoppingCriterion( StandardDeviationStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Computes and tests the simplex-value standard deviation.
   * @param[in] ordered_values Values ordered from best to worst.
   * @return Whether standard deviation is strictly below tolerance.
   * @pre At least two values are present.
   */
  bool shouldStop( const std::vector< Eigen::VectorXd > &, const std::vector< double > &ordered_values ) override
  {
    double mean = 0.0;
    for ( const double value : ordered_values )
    {
      mean += value;
    }
    mean /= static_cast< double >( ordered_values.size() );

    double squared_difference_sum = 0.0;
    for ( const double value : ordered_values )
    {
      const double difference = value - mean;
      squared_difference_sum += difference * difference;
    }

    const double simplex_dimension = static_cast< double >( ordered_values.size() - 1 );
    const double standard_deviation = std::sqrt( squared_difference_sum / simplex_dimension );
    return standard_deviation < configuration_.tolerance;
  }

  private:
  /** Copied strict standard-deviation threshold. */
  StandardDeviationStoppingConfiguration configuration_;
};

/*
 * Difference between the highest and lowest function values:
 *
 * f(x_H) <= f(x_L) + convergence_tolerance
 * or
 * f(x_L) <= absolute_tolerance
 *
 * Source: the Nelder-Mead termination condition used by R optim().
 */
/**
 * @brief Tests objective height or an internal absolute target.
 *
 * The result is true when \f$f_h\leq f_l+\epsilon\f$ or
 * \f$f_l\leq f_{target}\f$, using best \f$f_l\f$ and worst \f$f_h\f$.
 */
class HeightDifferenceStoppingCriterion final : public NelderMeadStoppingStrategy
{
  public:
  /** @brief Stores height and target thresholds. @param[in] configuration Validated thresholds. */
  explicit HeightDifferenceStoppingCriterion( HeightDifferenceStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Tests the ordered best and worst values.
   * @param[in] ordered_values Values ordered from best to worst.
   * @return Whether either inclusive threshold is satisfied.
   */
  bool shouldStop( const std::vector< Eigen::VectorXd > &, const std::vector< double > &ordered_values ) override
  {
    const double lowest_value = ordered_values.front();
    const double highest_value = ordered_values.back();

    return highest_value <= lowest_value + configuration_.convergence_tolerance ||
      lowest_value <= configuration_.absolute_tolerance;
  }

  private:
  /** Copied height and target thresholds. */
  HeightDifferenceStoppingConfiguration configuration_;
};

/*
 * Robust relative change between consecutive best function values:
 *
 * |f_{k+1} - f_k| / max(1, |f_k|) < ftol_rel
 *
 * Source: inspired by the NLopt relative function-tolerance criterion.
 */
/**
 * @brief Tests best-value change scaled by the previous best magnitude.
 *
 * After the first call, the value
 * \f$|f_k-f_{k-1}|/\max(1,|f_{k-1}|)\f$ is compared strictly with the
 * configured tolerance.
 */
class RelativeFunctionToleranceStoppingCriterion final : public NelderMeadStoppingStrategy
{
  public:
  /** @brief Stores the relative tolerance. @param[in] configuration Validated tolerance. */
  explicit RelativeFunctionToleranceStoppingCriterion( RelativeFunctionToleranceStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Updates the previous-best state and tests scaled change.
   * @param[in] ordered_values Values ordered from best to worst.
   * @return `false` on the first call; subsequently the strict tolerance result.
   */
  bool shouldStop( const std::vector< Eigen::VectorXd > &, const std::vector< double > &ordered_values ) override
  {
    const double current_best_value = ordered_values.front();
    if ( !previous_best_value_.has_value() )
    {
      previous_best_value_ = current_best_value;
      return false;
    }

    const double difference = std::abs( current_best_value - *previous_best_value_ );
    const double scale = std::max( 1.0, std::abs( *previous_best_value_ ) );
    previous_best_value_ = current_best_value;

    return difference / scale < configuration_.relative_tolerance;
  }

  private:
  /** Copied strict relative threshold. */
  RelativeFunctionToleranceStoppingConfiguration configuration_;
  /** Best mean from the preceding complete simplex. */
  std::optional< double > previous_best_value_;
};

/** @} */

#endif // !OPTIMIZER_NELDER_MEAD_STOPPING_CRITERION_H
