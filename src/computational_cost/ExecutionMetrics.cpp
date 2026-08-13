#include "ExecutionMetrics.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace
{
  std::uint64_t checkedAdd( std::uint64_t left, std::uint64_t right )
  {
    if ( right > std::numeric_limits< std::uint64_t >::max() - left )
    {
      throw std::overflow_error( "ExecutionMetrics: counter addition overflow" );
    }
    return left + right;
  }

  std::uint64_t checkedMultiply( std::uint64_t left, std::uint64_t right )
  {
    if ( left != 0 && right > std::numeric_limits< std::uint64_t >::max() / left )
    {
      throw std::overflow_error( "ExecutionMetrics: counter multiplication overflow" );
    }
    return left * right;
  }
} // namespace

void ExecutionMetrics::addEpisode( std::uint64_t simulation_steps )
{
  const std::uint64_t updated_episode_count = checkedAdd( number_of_episodes, 1 );
  const std::uint64_t updated_simulation_steps = checkedAdd( number_of_simulation_steps, simulation_steps );

  number_of_episodes = updated_episode_count;
  number_of_simulation_steps = updated_simulation_steps;
}

void ExecutionMetrics::addPolicyInferences( std::uint64_t inference_count, std::uint64_t macs_per_inference )
{
  const std::uint64_t updated_inference_count = checkedAdd( number_of_policy_inferences, inference_count );
  const std::uint64_t added_macs = checkedMultiply( inference_count, macs_per_inference );
  const std::uint64_t updated_macs = checkedAdd( number_of_neural_network_macs, added_macs );

  number_of_policy_inferences = updated_inference_count;
  number_of_neural_network_macs = updated_macs;
}

void ExecutionMetrics::append( const ExecutionMetrics &other )
{
  const std::uint64_t updated_episode_count = checkedAdd( number_of_episodes, other.number_of_episodes );
  const std::uint64_t updated_simulation_steps =
    checkedAdd( number_of_simulation_steps, other.number_of_simulation_steps );
  const std::uint64_t updated_inference_count =
    checkedAdd( number_of_policy_inferences, other.number_of_policy_inferences );
  const std::uint64_t updated_macs =
    checkedAdd( number_of_neural_network_macs, other.number_of_neural_network_macs );

  number_of_episodes = updated_episode_count;
  number_of_simulation_steps = updated_simulation_steps;
  number_of_policy_inferences = updated_inference_count;
  number_of_neural_network_macs = updated_macs;
}
