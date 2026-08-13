#include "ModelSelectionStoppingCriterionFactory.hpp"
#include <memory>
#include <type_traits>
#include <variant>
#include "ModelSelectionStoppingConfiguration.hpp"
#include "ModelSelectionStoppingStrategy.hpp"

std::unique_ptr< ModelSelectionStoppingStrategy >
ModelSelectionStoppingCriterionFactory::create( const ModelSelectionStoppingConfiguration &configuration )
{
  return std::visit(
    []( const auto &selected_configuration ) -> std::unique_ptr< ModelSelectionStoppingStrategy >
    {
      using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

      if constexpr ( std::is_same_v< ConfigurationType, ModelSelectionBudgetCriterionConfiguration > )
      {
        return std::make_unique< ModelSelectionBudgetCriterion >( selected_configuration );
      }
      else if constexpr ( std::is_same_v< ConfigurationType, ModelSelectionNoImprovementCriterionConfiguration > )
      {
        return std::make_unique< ModelSelectionNoImprovementCriterion >( selected_configuration );
      }
      else
      {
        return std::make_unique< ModelSelectionTargetScoreCriterion >( selected_configuration );
      }
    },
    configuration );
}
