#ifndef CONFIGURATION_AXIS_H
#define CONFIGURATION_AXIS_H

/**
 * @defgroup model_selection_api Model Selection
 * @brief Configuration spaces, selectors, validation protocols, scoring state, and results.
 * @{
 */

#include <stdexcept>
#include <utility>

/**
 * @brief Binds one search domain to a mutable configuration field.
 * @tparam Domain Domain type exposing `ValueType` and membership data.
 *
 * The axis stores a non-owning pointer. The pointed field must outlive the axis
 * and every selector that stores it. Moving the enclosing configuration would
 * invalidate that pointer.
 *
 * @see ConfigurationSearchSpace
 * @see model_selection_chapter
 */
template< typename Domain >
class ConfigurationAxis
{
  public:
  using ValueType = typename Domain::ValueType; ///< Type of the bound configuration field.

  /**
   * @brief Associates a configuration field with a search domain.
   * @param[in,out] parameter Non-owning pointer to the field modified by selectors.
   * @param[in] domain Domain copied into the axis.
   * @throws std::invalid_argument If `parameter` is null.
   */
  ConfigurationAxis( ValueType *parameter, Domain domain ) : parameter_( parameter ), domain_( std::move( domain ) )
  {
    if ( parameter_ == nullptr )
    {
      throw std::invalid_argument( "ConfigurationAxis: parameter cannot be null" );
    }
  }

  /** @brief Accesses the current field value. @return Mutable reference to the bound field. */
  ValueType &value() const { return *parameter_; }

  /**
   * @brief Assigns a value to the bound field.
   * @param[in] value Value to copy; domain membership is the caller's responsibility.
   * @post The pointed configuration field equals `value`.
   */
  void setValue( const ValueType &value ) const { *parameter_ = value; }

  /** @brief Returns the axis domain. @return Const reference valid for the axis lifetime. */
  const Domain &domain() const { return domain_; }
  /** @brief Returns the identity used to detect duplicate axes. @return Address of the bound field. */
  const void *address() const { return parameter_; }

  private:
  ValueType *parameter_; ///< Non-owning pointer to the configuration field.
  Domain domain_; ///< Owned domain definition.
};

/** @brief Deduces the domain type of an axis from its constructor arguments. */
template< typename Domain >
ConfigurationAxis( typename Domain::ValueType *, Domain ) -> ConfigurationAxis< Domain >;

/** @} */

#endif // !CONFIGURATION_AXIS_H
