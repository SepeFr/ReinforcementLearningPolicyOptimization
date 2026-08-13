#include "StoppingCriterionFactory.hpp"
#include <memory>
#include <type_traits>
#include <variant>
#include "OptimizationStoppingStrategy.hpp"
#include "StoppingConfiguration.hpp"

std::unique_ptr< OptimizationStoppingStrategy >
StoppingCriterionFactory::create( const StoppingConfiguration &configuration, OptimizationDirection direction )
{
  return std::visit(
    [direction]( const auto &selected_configuration ) -> std::unique_ptr< OptimizationStoppingStrategy >
    {
      using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

      if constexpr ( std::is_same_v< ConfigurationType, MaximumIterationsStoppingConfiguration > )
      {
        return std::make_unique< MaximumIterationsStoppingCriterion >( selected_configuration );
      }
      else if constexpr ( std::is_same_v< ConfigurationType, MaximumEvaluationsStoppingConfiguration > )
      {
        return std::make_unique< MaximumEvaluationsStoppingCriterion >( selected_configuration );
      }
      else if constexpr ( std::is_same_v< ConfigurationType, NoImprovementStoppingConfiguration > )
      {
        return std::make_unique< NoImprovementStoppingCriterion >( selected_configuration );
      }
      else
      {
        TargetValueStoppingConfiguration normalized_configuration = selected_configuration;
        if ( direction == OptimizationDirection::Maximize )
        {
          normalized_configuration.target_value = -normalized_configuration.target_value;
        }
        return std::make_unique< TargetValueStoppingCriterion >( normalized_configuration );
      }
    },
    configuration );
}
