#ifndef POLICY_SERIALIZATION_H
#define POLICY_SERIALIZATION_H

/** @addtogroup neural_network_api
 * @{ */

#include <eigen3/Eigen/Core>
#include <filesystem>
#include <utility>
#include "PolicyConfiguration.hpp"

/**
 * @brief Reads and writes the versioned text representation of a policy.
 *
 * Version 1 stores one comma-separated row for each configuration field, then
 * a `parameters` row in FeedForwardNetwork canonical order. Numeric output uses
 * enough decimal digits for a double round trip.
 */
class PolicySerialization
{
  public:
  using DeserializedPolicy = std::pair< PolicyConfiguration, Eigen::VectorXd >; ///< Configuration and matching parameters.

  /** @brief Static utility class; instances cannot be constructed. */
  PolicySerialization() = delete;

  /**
   * @brief Validates and writes one policy file.
   * @param[in] path Destination file, replaced if it already exists.
   * @param[in] configuration Policy configuration to store.
   * @param[in] parameters Canonical parameters matching @p configuration.
   * @throws InvalidConfigurationError If the policy configuration or parameter count is invalid.
   * @throws std::invalid_argument If a stored numeric value is non-finite or an enum value is unsupported.
   * @throws std::runtime_error If the file cannot be opened or completely written.
   */
  static void serialize( const std::filesystem::path &path, const PolicyConfiguration &configuration,
                         Eigen::Ref< const Eigen::VectorXd > parameters );

  /**
   * @brief Reads and validates a version 1 policy file.
   * @param[in] path Source file.
   * @return Parsed configuration and canonical parameter vector.
   * @throws std::runtime_error If the file cannot be opened, has missing,
   * malformed, reordered, unsupported, or trailing data, or contains a
   * non-finite numeric value.
   * @throws InvalidConfigurationError If parsed values violate policy constraints.
   */
  static DeserializedPolicy deserialize( const std::filesystem::path &path );
};

/** @} */

#endif // !POLICY_SERIALIZATION_H
