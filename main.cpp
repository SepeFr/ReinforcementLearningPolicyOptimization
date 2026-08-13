#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <utility>
#include <vector>
#include "exps/environments/PendulumStochastic.hpp"
#include "computational_cost/ComputationCostUpperBoundsCalculator.hpp"
#include "computational_cost/ConfigurationScoringConfiguration.hpp"
#include "model_selection/ConfigurationAxis.hpp"
#include "model_selection/ConfigurationSearchSpace.hpp"
#include "model_selection/ModelSelection.hpp"
#include "model_selection/ModelSelectionStoppingConfiguration.hpp"
#include "model_selection/ValidationProtocol.hpp"
#include "model_selection/configuration_selector/GridConfigurationSelector.hpp"
#include "model_selection/domains/DiscreteDomain.hpp"
#include "model_selection/model_selector/ModelSelectionFunctions.hpp"
#include "model_selection/model_selector/ModelSelectionResult.hpp"
#include "neural_networks/PolicyConfiguration.hpp"
#include "optimizer/InitializationConfiguration.hpp"
#include "optimizer/MethodHyperparameters.hpp"
#include "optimizer/OptimizerConfiguration.hpp"
#include "optimizer/StoppingConfiguration.hpp"

namespace
{
  using Scenario = exps::pendulum::Scenario;
  using Observation = exps::pendulum::Observation;
  using Action = exps::pendulum::Action;
  using StochasticPendulum = exps::pendulum::StochasticPendulumEnvironment;
  using Protocol = ValidationProtocol< Scenario, Observation, Action, StochasticPendulum >;

  std::vector< Scenario > makeScenarios( std::size_t count, std::uint64_t seed )
  {
    std::mt19937_64 generator( seed );
    std::uniform_real_distribution< double > angle_distribution(
      -std::numbers::pi_v< double >, std::numbers::pi_v< double > );
    std::uniform_real_distribution< double > velocity_distribution( -1.0, 1.0 );
    std::vector< Scenario > scenarios;
    scenarios.reserve( count );

    for ( std::size_t index = 0; index < count; ++index )
    {
      scenarios.push_back( Scenario{ angle_distribution( generator ), velocity_distribution( generator ) } );
    }
    return scenarios;
  }

  ModelConfiguration makeConfiguration()
  {
    PolicyConfiguration policy;
    policy.type = PolicyType::FeedForwardEigen;
    policy.network.input_size = 3;
    policy.network.hidden_layers = { 8 };
    policy.network.output_size = 1;
    policy.network.hidden_activation = ActivationType::Tanh;
    policy.network.output_activation = ActivationType::Tanh;
    policy.network.use_bias = true;
    policy.regularization_strategy = RegularizationType::L2_Regularization;
    policy.regularization_coefficient = 1e-6;

    CMAESHyperparameters cma_es;
    cma_es.initial_sigma = 0.5;
    cma_es.population_size = 32;
    cma_es.random_seed = 42;

    FeedForwardInitializationConfiguration initialization;
    initialization.network = policy.network;
    initialization.type = FeedForwardInitializationType::XavierUniform;
    initialization.random_seed = 42;

    OptimizerConfiguration optimizer;
    optimizer.hyperparameters = cma_es;
    optimizer.initialization = initialization;
    optimizer.stopping = { MaximumIterationsStoppingConfiguration{ 30 } };
    optimizer.random_seed = 42;

    return ModelConfiguration{ std::move( optimizer ), std::move( policy ) };
  }
}

int main()
{
  try
  {
    StochasticPendulum environment( 2026, 0.2 );
    ValidationProtocolConfiguration< Scenario, StochasticPendulum > validation{
      makeScenarios( 6, 1001 ), makeScenarios( 4, 1002 ), std::ref( environment ), 3, 3
    };

    ModelConfiguration configuration = makeConfiguration();
    CMAESHyperparameters &cma_es =
      std::get< CMAESHyperparameters >( configuration.optimizer_configuration.hyperparameters );
    auto sigma_axis = ConfigurationAxis(
      &cma_es.initial_sigma, DiscreteDomain< double >( { 0.5, 0.25, 0.1 } ) );
    auto population_axis = ConfigurationAxis(
      &cma_es.population_size, DiscreteDomain< std::size_t >( { 32, 16 } ) );
    auto search_space = ConfigurationSearchSpace( sigma_axis, population_axis );
    using Selector = GridConfigurationSelector< decltype( sigma_axis ), decltype( population_axis ) >;

    auto selector = std::make_unique< Selector >(
      configuration, std::move( search_space ), 1.0 );
    ModelSelectionFunctions functions = makeModelSelectionFunctions< Protocol >(
      validation, makeScenarios( 6, 1003 ) );
    std::vector< ModelSelectionStoppingConfiguration > stopping{
      ModelSelectionBudgetCriterionConfiguration{ 6 }
    };

    ConfigurationScoringConfiguration scoring;
    scoring.simulation_cost_weight = 1.0;
    ComputationCostUpperBoundInputs bound_inputs;
    bound_inputs.maximum_black_box_evaluations = 960;
    bound_inputs.maximum_episode_steps = 200;
    ValidationConfiguration< Scenario, StochasticPendulum > bounded_validation = validation;
    const ComputationCostUpperBounds upper_bounds =
      ComputationCostUpperBoundsCalculator::calculate< Scenario, StochasticPendulum >(
        scoring, bounded_validation, bound_inputs );

    std::cout << "Stochastic Pendulum demo: 3-8-1 Tanh policy, CMA-ES, 30 iterations\n";
    std::cout << "Model selection: sigma {0.10, 0.25, 0.50}, population {16, 32}\n";
    std::cout << "Selection score includes a small normalized simulation-cost penalty\n";

    ModelSelection selection(
      std::move( selector ), std::move( functions ), std::move( stopping ),
      scoring, upper_bounds,
      FeedForwardNetworkConfiguration::parameterCountFromConfiguration( configuration.policy.network ) );
    const ModelSelectionResult result = selection.select();

    std::cout << std::fixed << std::setprecision( 2 );
    for ( const ModelSelectionRecord &record : result.evaluated_configurations )
    {
      const CMAESHyperparameters &candidate = std::get< CMAESHyperparameters >(
        record.configuration.optimizer_configuration.hyperparameters );
      std::cout << "  sigma=" << candidate.initial_sigma
                << ", population=" << candidate.population_size
                << " -> validation return=" << record.evaluation.validation_mean
                << ", score=" << *record.evaluation.selection_score << '\n';
    }

    const CMAESHyperparameters &best = std::get< CMAESHyperparameters >(
      result.best_configuration.optimizer_configuration.hyperparameters );
    std::cout << "Selected: sigma=" << best.initial_sigma
              << ", population=" << best.population_size << '\n';
    std::cout << "Final test return: " << result.test_evaluation.meanValue()
              << " +/- " << result.test_evaluation.standardDeviation() << '\n';
    std::cout << "Final training: " << result.final_training_result.number_of_iterations
              << " iterations, " << result.final_training_result.number_of_evaluations
              << " evaluations\n";
    return EXIT_SUCCESS;
  }
  catch ( const std::exception &error )
  {
    std::cerr << "Demo failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
