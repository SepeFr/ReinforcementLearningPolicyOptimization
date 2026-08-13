#ifndef DOMAIN_KIND_H
#define DOMAIN_KIND_H

/** @addtogroup model_selection_api
 * @{ */

/** @file DomainKind.hpp @brief Configuration-domain sampling categories. */

/** @brief Identifies the sampling and enumeration semantics of a configuration domain. */
enum class DomainKind
{
  Discrete, ///< Selects values from a stored sequence.
  Continuous, ///< Selects floating-point values from a closed interval.
  Categorical ///< Selects named values from a stored sequence.
};

/** @} */

#endif // !DOMAIN_KIND_H
