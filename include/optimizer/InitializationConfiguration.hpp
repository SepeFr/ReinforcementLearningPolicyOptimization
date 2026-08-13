#ifndef INITIALIZATION_CONFIGURATION_H
#define INITIALIZATION_CONFIGURATION_H

/** @addtogroup optimizer_core_api
 * @{ */

/** @file InitializationConfiguration.hpp @brief Optimizer initial-parameter configuration alternatives. */

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <optional>
#include <variant>
#include <vector>
#include "neural_networks/FeedForwardNetworkConfiguration.hpp"

struct RandomInitializationConfiguration;
struct FeedForwardInitializationConfiguration;
struct ProvidedInitializationConfiguration;
struct ProvidedInitializationVectorConfiguration;

/** @brief Variant selecting one initial-parameter generation strategy. */
using InitializationConfiguration =
  std::variant< RandomInitializationConfiguration, FeedForwardInitializationConfiguration,
                ProvidedInitializationConfiguration, ProvidedInitializationVectorConfiguration >;

/** @brief Configures independent random coordinates for initial candidates. */
struct RandomInitializationConfiguration
{
  std::uint64_t random_seed = 0; ///< Deterministic seed restored by reset().
  std::optional< Eigen::VectorXd > random_lower_bound; ///< Optional per-coordinate inclusive lower limits.
  std::optional< Eigen::VectorXd > random_upper_bound; ///< Optional per-coordinate inclusive upper limits.
};

/** @brief Uniform weight interval derived from adjacent layer widths. */
enum class FeedForwardInitializationType
{
  FanInUniform, ///< Samples weights from \f$[-1/\sqrt{n_{in}},1/\sqrt{n_{in}}]\f$.
  XavierUniform ///< Samples weights from \f$[-\sqrt{6/(n_{in}+n_{out})},\sqrt{6/(n_{in}+n_{out})}]\f$.
};

/** @brief Configures layer-aware random initialization of a feed-forward network. */
struct FeedForwardInitializationConfiguration
{
  FeedForwardNetworkConfiguration network; ///< Network whose canonical parameter layout is initialized.
  FeedForwardInitializationType type = FeedForwardInitializationType::XavierUniform; ///< Weight-bound formula.
  std::uint64_t random_seed = 0; ///< Deterministic seed restored by reset().
};

/** @brief Configures one fixed initial candidate. */
struct ProvidedInitializationConfiguration
{
  Eigen::VectorXd provided_parameters; ///< Nonempty candidate returned on every request.
};

/** @brief Configures a cyclic sequence of fixed initial candidates. */
struct ProvidedInitializationVectorConfiguration
{
  std::vector< Eigen::VectorXd > provided_parameters; ///< Nonempty sequence returned in stored order.
};


/** @} */

#endif // !INITIALIZATION_CONFIGURATION_H
