#ifndef HILL_CLIMBING_METHOD_H
#define HILL_CLIMBING_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cassert>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <utility>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "MethodHyperparameters.hpp"
#include "NeighborhoodStrategy.hpp"
#include "OptimizerMethod.hpp"

class OptimizerFactory;

/**
 * @brief Replaces a current point with the best accepted candidate in a local batch.
 *
 * The first batch contains one initial candidate. Later batches contain
 * number_of_neighbors scaled perturbations. The lowest-mean successful neighbor
 * replaces the current point on strict improvement, or on exact equality when
 * configured.
 *
 * @see optimization_method_states_chapter
 */
class HillClimbingMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Produces the initial point or a batch around the current point.
   * @return One initial candidate before initialization; otherwise exactly
   * HillClimbingHyperparameters::number_of_neighbors scaled neighbors.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Clears the current point and resets the owned neighborhood. */
  void reset() override;

  protected:
  /**
   * @brief Accepts initialization feedback or selects the best local neighbor.
   * @param[in] evaluations Feedback in the order returned by ask().
   * @return A completed logical iteration for every accepted feedback batch.
   * @throws std::out_of_range If initial feedback is empty.
   * @throws ConfigurationEvaluationError If the initial candidate failed.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs the method with default hyperparameters and Gaussian neighborhood.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   */
  explicit HillClimbingMethod( InitialParametersStrategy *initial_parameters_strategy ) :
      HillClimbingMethod( HillClimbingHyperparameters{}, std::make_unique< GaussianNeighborhood >(),
                          initial_parameters_strategy )
  {
  }

  /**
   * @brief Takes ownership of the configured neighborhood.
   * @param[in] hyperparameters Batch, scale, and equality settings copied into the method.
   * @param[in] neighborhood Non-null owned displacement strategy.
   * @param[in] initial_parameters_strategy Non-owning strategy that outlives the method's use of it.
   */
  HillClimbingMethod( const HillClimbingHyperparameters &hyperparameters,
                      std::unique_ptr< NeighborhoodStrategy > neighborhood,
                      InitialParametersStrategy *initial_parameters_strategy ) :
      OptimizerMethod( initial_parameters_strategy ), neighborhood_( std::move( neighborhood ) ),
      hyperparameters_( hyperparameters )
  {
    assert( neighborhood_ && "HillClimbingMethod: neighborhood cannot be null" );
    assert( hyperparameters_.number_of_neighbors > 0 &&
            "HillClimbingMethod: number of neighbors must be greater than zero" );
  }

  /** Owned local displacement strategy. */
  std::unique_ptr< NeighborhoodStrategy > neighborhood_;
  /** Copied scale, batch, and acceptance settings. */
  HillClimbingHyperparameters hyperparameters_;
  /** Accepted current point and evaluation. */
  CandidateEvaluation current_candidate_;
  /** Distinguishes the initial-candidate phase from local iterations. */
  bool has_current_candidate_ = false;
};

/** @} */

#endif // !HILL_CLIMBING_METHOD_H
