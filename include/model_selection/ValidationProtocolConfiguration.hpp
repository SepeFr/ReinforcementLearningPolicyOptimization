#ifndef VALIDATION_PROTOCOL_CONFIGURATION_H
#define VALIDATION_PROTOCOL_CONFIGURATION_H

/** @addtogroup model_selection_api
 * @{ */

/** @file ValidationProtocolConfiguration.hpp @brief Holdout and K-fold protocol configuration types. */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <variant>
#include <vector>

/**
 * @brief Configures training and holdout validation over explicit scenario sets.
 * @tparam ScenarioType Scenario value copied into problem instances.
 * @tparam EnvironmentType Environment shared by training and validation episodes.
 *
 * The environment reference is non-owning and must remain valid throughout
 * model selection. Training and validation use it sequentially.
 */
template< typename ScenarioType, typename EnvironmentType >
struct ValidationProtocolConfiguration
{
  std::vector< ScenarioType > training_scenarios; ///< Non-empty scenarios used in every objective evaluation during training.
  std::vector< ScenarioType > validation_scenarios; ///< Non-empty disjoint-role scenarios used after training.
  std::reference_wrapper< EnvironmentType > environment; ///< Non-owning reference to the resettable environment instance.
  std::size_t training_runs_per_scenario = 50; ///< Positive episode replicas per training scenario and candidate.
  std::size_t validation_runs_per_scenario = 50; ///< Positive episode replicas per validation or test scenario.
};

/**
 * @brief Extends validation settings with deterministic K-fold partitioning.
 * @tparam ScenarioType Scenario value copied into fold vectors.
 * @tparam EnvironmentType Environment shared across all folds.
 */
template< typename ScenarioType, typename EnvironmentType >
struct KFoldCrossValidationConfiguration : ValidationProtocolConfiguration< ScenarioType, EnvironmentType >
{
  std::size_t fold_count = 5; ///< Number of folds; must be at least two and at most the combined scenario count.
  std::uint64_t random_seed = 42; ///< Seed used to reproduce the permutation of combined scenarios.
};

/**
 * @brief Selects holdout or K-fold validation configuration.
 * @tparam ScenarioType Scenario value type.
 * @tparam EnvironmentType Shared environment type.
 */
template< typename ScenarioType, typename EnvironmentType >
using ValidationConfiguration = std::variant< ValidationProtocolConfiguration< ScenarioType, EnvironmentType >,
                                              KFoldCrossValidationConfiguration< ScenarioType, EnvironmentType > >;

/** @} */

#endif // !VALIDATION_PROTOCOL_CONFIGURATION_H
