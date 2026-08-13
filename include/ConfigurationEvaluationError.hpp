#ifndef CONFIGURATION_EVALUATION_ERROR_H
#define CONFIGURATION_EVALUATION_ERROR_H

/** @addtogroup validation_api
 * @{ */

/** @file ConfigurationEvaluationError.hpp @brief Recoverable model-configuration evaluation failures. */

#include <stdexcept>
#include <string>

/** @brief Classifies failures that invalidate one model-configuration evaluation. */
enum class ConfigurationEvaluationErrorKind
{
  EvaluationFailure, ///< Training or validation could not produce an evaluation.
  NonFiniteValidationStatistics, ///< Aggregated validation mean or variance is invalid.
  NonFiniteSelectionScore ///< Performance and cost terms produced a non-finite score.
};

/**
 * @brief Reports a recoverable failure of one configuration evaluation.
 *
 * ModelSelection converts this exception into failed selector feedback while
 * allowing subsequent configurations to be evaluated.
 */
class ConfigurationEvaluationError : public std::runtime_error
{
  public:
  /**
   * @brief Creates an evaluation error with a machine-readable category.
   * @param[in] message Diagnostic message returned by `what()`.
   * @param[in] kind Failure category; defaults to a general evaluation failure.
   */
  explicit ConfigurationEvaluationError(
    const std::string &message,
    ConfigurationEvaluationErrorKind kind = ConfigurationEvaluationErrorKind::EvaluationFailure ) :
      std::runtime_error( message ), kind_( kind )
  {
  }

  /** @brief Returns the failure category. @return Category supplied to the constructor. */
  ConfigurationEvaluationErrorKind kind() const noexcept { return kind_; }

  private:
  ConfigurationEvaluationErrorKind kind_; ///< Stable category associated with this exception.
};

/** @} */

#endif // !CONFIGURATION_EVALUATION_ERROR_H
