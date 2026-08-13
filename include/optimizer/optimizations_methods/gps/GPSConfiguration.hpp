#ifndef OPTIMIZER_GPS_CONFIGURATION_H
#define OPTIMIZER_GPS_CONFIGURATION_H

/** @addtogroup optimization_methods_api
 * @{ */

/** @file GPSConfiguration.hpp @brief GPS basis, search, poll, and stochastic sampling configuration alternatives. */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <optional>
#include <variant>
#include <vector>
#include "optimizations_methods/gps/StochasticGPSStoppingCriterion.hpp"
#include "surrogate_models/SurrogateModelConfiguration.hpp"

/** @brief Selects integer generator columns \f$Z=[I,-\mathbf{1}]\f$. */
struct MinimalPositiveBasisConfiguration
{
};

/** @brief Selects integer generator columns \f$Z=[I,-I]\f$. */
struct SymmetricPositiveBasisConfiguration
{
};

/** @brief Variant selecting the positive-basis generator used by GPS polling. */
using GPSPositiveBasisConfiguration =
  std::variant< MinimalPositiveBasisConfiguration, SymmetricPositiveBasisConfiguration >;

/** @brief Disables the optional GPS search phase. */
struct EmptyGPSSearchConfiguration
{
};

/** @brief Configures independent uniform draws from a finite integer L1 mesh ball. */
struct RandomMeshGPSSearchConfiguration
{
  std::size_t number_of_points; ///< Search batch cardinality; zero disables this search.
  std::size_t l1_radius; ///< Positive maximum integer coefficient L1 norm when points are requested.
  std::size_t random_seed = 0; ///< Seed restored by strategy reset().
};

/** @brief Configures stratified integer directions within an L1 mesh ball. */
struct LatinHypercubeMeshGPSSearchConfiguration
{
  std::size_t number_of_points; ///< Target number of candidates; zero disables this search.
  std::size_t l1_radius; ///< Positive maximum integer coefficient L1 norm.
  std::size_t maximum_batches = 16; ///< Positive design batches attempted to reach the target.
  bool remove_duplicates = true; ///< Deduplicates integer coefficient vectors within one ask() call.
  std::size_t random_seed = 0; ///< Latin hypercube seed restored by reset().
};

/** @brief Configures extrapolation from the most recent improving mesh direction. */
struct SuccessfulDirectionGPSSearchConfiguration
{
  std::vector< int > multipliers; ///< Ordered integer multiples applied to the successful direction.
  RandomMeshGPSSearchConfiguration initial_search; ///< Search used until an improving direction is observed.
};

/** @brief Configures random mesh generation followed by surrogate ranking. */
struct SurrogateGPSSearchConfiguration
{
  RandomMeshGPSSearchConfiguration random_points_strategy; ///< Pool generator before model ranking.
  SurrogateModelConfiguration model; ///< Regression model created and owned by the search strategy.
  double selected_fraction = 0.25; ///< Finite selected share in `(0,1]`, rounded upward with at least one for a nonempty pool.
};

/** @brief Variant selecting the optional finite GPS search phase. */
using GPSSearchConfiguration =
  std::variant< EmptyGPSSearchConfiguration, RandomMeshGPSSearchConfiguration,
                LatinHypercubeMeshGPSSearchConfiguration, SuccessfulDirectionGPSSearchConfiguration,
                SurrogateGPSSearchConfiguration >;

/** @brief Selects every column of the complete GPS positive basis for polling. */
struct CompleteGPSPollConfiguration
{
};

/** @brief Variant selecting the GPS poll-direction subset. */
using GPSPollConfiguration = std::variant< CompleteGPSPollConfiguration >;

/** @brief Configures mesh scaling, directions, search, poll, and convergence for GPS. */
struct GeneralizedPatternSearchHyperparameters
{
  double initial_mesh_size = 1.0; ///< Finite positive initial \f$\delta_0\f$.
  double mesh_size_adjustment = 0.5; ///< Finite \f$\tau\in(0,1)\f$; failure multiplies and success divides mesh size by it.
  double stopping_mesh_size = 0.001; ///< Finite non-negative strict convergence threshold; zero disables the internal test.

  std::optional< Eigen::MatrixXd > generating_matrix; ///< Optional finite invertible square \f$G\f$; identity by default.
  GPSPositiveBasisConfiguration positive_basis = MinimalPositiveBasisConfiguration{}; ///< Integer \f$Z\f$ used in \f$D=GZ\f$.
  GPSSearchConfiguration search = EmptyGPSSearchConfiguration{}; ///< Optional candidates evaluated before polling.
  GPSPollConfiguration poll = CompleteGPSPollConfiguration{}; ///< Poll-direction selection strategy.
};

/** @brief Adds two-stage Rinott ranking-and-selection sampling to GPS. */
struct RinottRankingSelectionGPSHyperparameters
  : GeneralizedPatternSearchHyperparameters
{
  std::size_t initial_sample_size = 5; ///< First-stage replications per alternative; at least two.

  double initial_significance = 0.8; ///< Initial confidence probability strictly between zero and one.
  double initial_indifference = 1.0; ///< Finite positive indifference-zone width.

  double significance_decay = 0.95; ///< Iteration multiplier strictly between zero and one.
  double indifference_decay = 0.95; ///< Iteration multiplier strictly between zero and one.

  StochasticGPSStoppingConfiguration stopping_criterion =
    StochasticGPSMeshSizeStoppingConfiguration{}; ///< Stochastic convergence test applied after completed iterations.
};

/** @} */

#endif // !OPTIMIZER_GPS_CONFIGURATION_H
