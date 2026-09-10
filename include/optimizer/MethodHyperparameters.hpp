#ifndef METHOD_HYPERPARAMETERS_H
#define METHOD_HYPERPARAMETERS_H

/** @addtogroup optimization_methods_api
 * @{ */

/** @file MethodHyperparameters.hpp @brief Concrete optimizer-method hyperparameter alternatives. */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <variant>
#include "NeighborhoodConfiguration.hpp"
#include "optimizations_methods/gps/GPSConfiguration.hpp"
#include "optimizations_methods/nelder_mead/NelderMeadInitializationConfiguration.hpp"
#include "optimizations_methods/nelder_mead/NelderMeadStoppingCriterion.hpp"
#include "optimizations_methods/simulated_annealing/CoolingFunctionConfiguration.hpp"

struct HillClimbingHyperparameters;
struct SimulatedAnnealingHyperparameters;
struct NelderMeadHyperparameters;
struct StochasticHeuristicNelderMeadHyperparameters;
struct ChangStochasticNelderMeadHyperparameters;
struct RinottRankingSelectionGPSHyperparameters;
struct OpenAIESHyperparameters;
struct CMAESHyperparameters;
struct PSOHyperparameters;
struct SPSAHyperparameters;

/** @brief Variant selecting an optimization method and its algorithmic settings. */
using MethodHyperparameters =
  std::variant< HillClimbingHyperparameters, SimulatedAnnealingHyperparameters, NelderMeadHyperparameters,
                StochasticHeuristicNelderMeadHyperparameters, ChangStochasticNelderMeadHyperparameters,
                GeneralizedPatternSearchHyperparameters, RinottRankingSelectionGPSHyperparameters,
                OpenAIESHyperparameters, CMAESHyperparameters, PSOHyperparameters, SPSAHyperparameters >;

/** @brief Configures batched best-neighbor hill climbing. */
struct HillClimbingHyperparameters
{
  NeighborhoodConfiguration neighborhood = GaussianNeighborhoodConfiguration{}; ///< Base displacement generator.
  double neighborhood_scale = 0.1; ///< Finite positive multiplier applied to generated displacements.
  std::size_t number_of_neighbors = 1; ///< Positive candidates requested per local iteration.
  bool accept_equal_candidates = false; ///< Accepts an exactly equal best-neighbor mean when enabled.
};

/** @brief Configures single-proposal simulated annealing. */
struct SimulatedAnnealingHyperparameters
{
  NeighborhoodConfiguration neighborhood = GaussianNeighborhoodConfiguration{}; ///< Proposal displacement generator.
  CoolingFunctionConfiguration cooling_function = GeometricCoolingConfiguration{}; ///< Temperature schedule.
  double initial_temperature = 1.0; ///< Finite positive temperature supplied to the schedule.
  double minimal_temperature = 0.0001; ///< Non-negative strict convergence threshold.
  double neighborhood_scale = 0.1; ///< Finite positive multiplier applied to proposal displacement.
};

/** @brief Configures simplex construction, transitions, and convergence. */
struct NelderMeadHyperparameters
{
  NelderMeadStoppingConfiguration stopping_criterion = DispersionStoppingConfiguration{}; ///< Test applied to the sorted simplex.
  NelderMeadInitializationConfiguration initialization = ClassicalLocalSimplexConfiguration{}; ///< Simplex construction strategy.
  double initial_simplex_scale = 0.1; ///< Finite positive scale supplied to applicable initializers.
  double reflection_coefficient = 1.0; ///< Finite \f$\alpha\f$ in \f$c+\alpha(c-x_n)\f$.
  double expansion_coefficient = 2.0; ///< Finite \f$\gamma\f$ in \f$c+\gamma(c-x_n)\f$.
  double inside_contraction_coefficient = -0.5; ///< Finite signed \f$\rho_i\f$ in \f$c+\rho_i(c-x_n)\f$.
  double outside_contraction_coefficient = 0.5; ///< Finite \f$\rho_o\f$ in \f$c+\rho_o(c-x_n)\f$.
  double shrink_coefficient = 0.5; ///< Finite multiplier from the best vertex to each remaining vertex.
};

/** @brief Adds resampling choices to heuristic stochastic Nelder--Mead. */
struct StochasticHeuristicNelderMeadHyperparameters : NelderMeadHyperparameters
{
  bool reevaluate_stagnant_best = true; ///< Resamples an unchanged best after a completed base iteration.
  bool reevaluate_best_before_shrink = true; ///< Refreshes the best estimate before accepting a shrink batch.
  bool confirm_contraction = false; ///< Resamples the current worst before accepting a contraction replacement.

  /** @brief Constructs base defaults and changes shrink_coefficient to `0.75`. */
  StochasticHeuristicNelderMeadHyperparameters() { shrink_coefficient = 0.75; }
};

/** @brief Configures Chang's cumulative-replication stochastic simplex method. */
struct ChangStochasticNelderMeadHyperparameters : NelderMeadHyperparameters
{
  std::size_t minimum_sample_size = 1; ///< Positive lower bound for cumulative replications per vertex.
  double global_search_probability = 0.1; ///< Probability strictly between zero and one of proposing a global point.
  Eigen::VectorXd lower_bound; ///< Finite strict lower sampling limits in parameter order.
  Eigen::VectorXd upper_bound; ///< Finite strict upper sampling limits in parameter order.
  std::size_t random_seed = 42; ///< Seed restored for global/local proposals.
};

/** @brief Configures antithetic OpenAI evolution strategies. */
struct OpenAIESHyperparameters
{
  enum class UpdateMethod
  {
    SGD,
    Adam,
    AdamW
  };

  std::size_t population_size = 50; ///< Even population cardinality of at least two.
  double learning_rate = 0.01; ///< Finite positive mean-update multiplier.
  double noise_scale = 0.1; ///< Finite positive Gaussian smoothing scale \f$\sigma\f$.
  bool adapt_noise_scale = false; ///< Applies the one-fifth mutation-success rule when enabled.
  double target_success_rate = 0.2; ///< One-fifth-rule target probability of beating the parent center.
  double noise_scale_increase_factor = 1.05; ///< Multiplier used above the target success rate.
  double noise_scale_decrease_factor = 0.95; ///< Multiplier used below the target success rate.
  double minimum_noise_scale = 1e-6; ///< Positive lower clamp for adaptive \f$\sigma\f$.
  double maximum_noise_scale = 1.0; ///< Upper clamp for adaptive \f$\sigma\f$.
  UpdateMethod update_method = UpdateMethod::AdamW; ///< Mean-update algorithm.
  double beta_1 = 0.9; ///< Adam/AdamW first-moment decay.
  double beta_2 = 0.999; ///< Adam/AdamW second-moment decay.
  double epsilon = 1e-8; ///< Adam/AdamW denominator stability constant.
  double weight_decay = 1e-4; ///< AdamW's decoupled parameter-decay coefficient.
  bool use_rank_fitness = true; ///< Converts minimization values to centered ranks when enabled.
  std::size_t random_seed = 42; ///< Seed restored for Gaussian perturbations.
  /** Prefix held fixed during ask/update. Zero optimizes the complete vector. */
  std::size_t frozen_prefix_size = 0;
};

/** @brief Configures the population and initial global scale of CMA-ES. */
struct CMAESHyperparameters
{
  double initial_sigma = 1.0; ///< Finite positive initial global step size \f$\sigma_0\f$.
  std::size_t population_size = 100; ///< Generation size \f$\lambda\geq2\f$.
  std::size_t random_seed = 42; ///< Seed restored for standard-normal samples.
  std::size_t eigendecomposition_period = 10; ///< Generations between eigensystem refreshes.
};

/** @brief Configures particle-swarm velocity and population updates. */
struct PSOHyperparameters
{
  double initial_inertia_weight = 0.7298; ///< Finite non-negative previous-velocity coefficient.
  double cognitive_coefficient = 1.49618; ///< Finite non-negative personal-best attraction coefficient.
  double social_coefficient = 1.49618; ///< Finite non-negative global-best attraction coefficient.
  double initial_velocity_scale = 0.1; ///< Finite positive half-width of initial uniform velocities.
  std::size_t population_size = 100; ///< Positive number of particles and candidates per generation.
  std::size_t random_seed = 42; ///< Seed restored for velocities and attraction factors.
};

/** @brief Configures simultaneous perturbation stochastic approximation. */
struct SPSAHyperparameters
{
  double perturbation_magnitude = 0.1; ///< Finite positive symmetric perturbation scale \f$c\f$.
  double step_size = 0.01; ///< Finite positive gradient step multiplier \f$a\f$.
  std::size_t random_seed = 42; ///< Seed restored for Rademacher directions.
};


/** @} */

#endif // !METHOD_HYPERPARAMETERS_H
