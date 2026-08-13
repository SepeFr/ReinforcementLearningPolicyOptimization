#ifndef POLICY_CONFIGURATION_H
#define POLICY_CONFIGURATION_H

/** @addtogroup neural_network_api
 * @{ */

/** @file PolicyConfiguration.hpp @brief Policy implementation and regularization configuration. */

#include "FeedForwardNetworkConfiguration.hpp"
/** @brief Available parametrized-policy implementations. */
enum class PolicyType
{
  FeedForwardEigen ///< Eigen-based FeedForwardPolicy implementation.
};

/** @brief Penalty applied to policy weights during optimization scoring. */
enum class RegularizationType
{
  L2_Regularization, ///< Squared Euclidean weight penalty.
  L1_Regularization ///< Absolute-value weight penalty.
};

/** @brief Selects a policy implementation, network, and weight regularization. */
struct PolicyConfiguration
{
  PolicyType type; ///< Concrete policy implementation created by PolicyFactory.
  FeedForwardNetworkConfiguration network; ///< Feed-forward network structure.
  RegularizationType regularization_strategy = RegularizationType::L2_Regularization; ///< Norm used by PolicyOptimizationProblem.
  double regularization_coefficient = 0.0; ///< Non-negative penalty multiplier; zero disables its contribution.
};


/** @} */

#endif // !POLICY_CONFIGURATION_H
