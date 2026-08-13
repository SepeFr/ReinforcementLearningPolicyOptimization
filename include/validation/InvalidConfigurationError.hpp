#ifndef INVALID_CONFIGURATION_ERROR_H
#define INVALID_CONFIGURATION_ERROR_H

/**
 * @defgroup validation_api Validation and Errors
 * @brief Configuration validators and shared configuration-evaluation errors.
 * @{
 */

#include <stdexcept>

/**
 * @brief Reports a configuration value or cross-component invariant rejected before execution.
 *
 * ModelSelection records this exception as
 * ModelSelectionFailureKind::InvalidConfiguration for the current attempt.
 */
class InvalidConfigurationError : public std::invalid_argument
{
  public:
  /** @brief Inherits message-based constructors from `std::invalid_argument`. */
  using std::invalid_argument::invalid_argument;
};

/** @} */

#endif // !INVALID_CONFIGURATION_ERROR_H
