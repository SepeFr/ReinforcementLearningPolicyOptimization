#ifndef OPTIMIZER_GPS_SEARCH_STRATEGY_H
#define OPTIMIZER_GPS_SEARCH_STRATEGY_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <random>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "LatinHypercubeSampler.hpp"
#include "optimizations_methods/gps/GPSConfiguration.hpp"
#include "surrogate_models/SurrogateModel.hpp"

/**
 * @brief Interface for finite candidate generation before a GPS poll.
 *
 * GPS supplies the incumbent, current mesh size, and complete direction matrix.
 * tell() receives both search and poll observations so stateful strategies can
 * learn from the complete iteration history.
 */
class GPSSearchStrategy
{
  public:
  /** @brief Enables destruction through the search interface. */
  virtual ~GPSSearchStrategy() = default;

  /**
   * @brief Generates a finite ordered search batch on the current mesh.
   * @param[in] current_candidate Successful incumbent in minimization space.
   * @param[in] mesh_size Positive current \f$\delta_k\f$.
   * @param[in] directions Complete matrix \f$D\f$ used to map integer coefficients.
   * @param[in] iteration Completed GPS iteration count \f$k\f$.
   * @return Search candidates; an empty vector proceeds directly to poll.
   */
  virtual std::vector< Eigen::VectorXd > ask( const CandidateEvaluation &current_candidate, double mesh_size,
                                              Eigen::Ref< const Eigen::MatrixXd > directions,
                                              std::size_t iteration ) = 0;

  /**
   * @brief Observes a complete search or poll feedback batch before GPS state changes.
   * @param[in] current_candidate Incumbent used to generate the trials.
   * @param[in] evaluations Trial evaluations in request order.
   * @param[in] mesh_size Mesh size used to generate the trials.
   */
  virtual void tell( const CandidateEvaluation &current_candidate,
                     const std::vector< CandidateEvaluation > &evaluations, double mesh_size ) = 0;
  /** @brief Restores deterministic and learned strategy state. */
  virtual void reset() = 0;
};

/** @brief Produces no search candidates, causing GPS to poll immediately. */
class EmptyGPSSearchStrategy : public GPSSearchStrategy
{
  public:
  /** @brief Stores the marker configuration. @param[in] configuration Empty-search marker. */
  explicit EmptyGPSSearchStrategy( EmptyGPSSearchConfiguration configuration );

  /** @copydoc GPSSearchStrategy::ask() */
  std::vector< Eigen::VectorXd > ask( const CandidateEvaluation &current_candidate, double mesh_size,
                                      Eigen::Ref< const Eigen::MatrixXd > directions, std::size_t iteration ) override
  {
    static_cast< void >( current_candidate );
    static_cast< void >( mesh_size );
    static_cast< void >( directions );
    static_cast< void >( iteration );
    return {};
  }

  /** @copydoc GPSSearchStrategy::tell() */
  void tell( const CandidateEvaluation &current_candidate, const std::vector< CandidateEvaluation > &evaluations,
             double mesh_size ) override
  {
    static_cast< void >( current_candidate );
    static_cast< void >( evaluations );
    static_cast< void >( mesh_size );
  }
  /** @brief Performs no work because the strategy is stateless. */
  void reset() override {}

  private:
  /** Stored empty-search marker. */
  EmptyGPSSearchConfiguration configuration_;
};

/**
 * @brief Samples integer coefficient vectors uniformly from a bounded L1 ball.
 *
 * Each nonzero \f$y\in\mathbb Z^p\f$ with
 * \f$\lVert y\rVert_1\leq L\f$ has equal probability. Candidates are
 * \f$x_k+\delta_k D y\f$. Sampling uses replacement, so duplicates may occur.
 */
class RandomMeshGPSSearchStrategy : public GPSSearchStrategy
{
  public:
  /**
   * @brief Stores mesh-ball settings and seeds the generator.
   * @param[in] configuration Point count, L1 radius, and seed.
   * @throws std::invalid_argument If points are requested with a zero radius.
   */
  explicit RandomMeshGPSSearchStrategy( RandomMeshGPSSearchConfiguration configuration );

  /** @copydoc GPSSearchStrategy::ask() */
  std::vector< Eigen::VectorXd > ask( const CandidateEvaluation &current_candidate, double mesh_size,
                                      Eigen::Ref< const Eigen::MatrixXd > directions, std::size_t iteration ) override;

  /** @copydoc GPSSearchStrategy::tell() */
  void tell( const CandidateEvaluation &current_candidate, const std::vector< CandidateEvaluation > &evaluations,
             double mesh_size ) override;
  /** @brief Reseeds the random engine with the configured seed. */
  void reset() override;

  private:
  /** Copied point count, radius, and seed. */
  RandomMeshGPSSearchConfiguration configuration_;
  /** Random engine for coefficient shape, support, magnitudes, and signs. */
  std::mt19937_64 generator_;
};

/**
 * @brief Projects stratified integer coefficients onto a bounded L1 mesh ball.
 *
 * Continuous Latin hypercube coordinates are mapped to integer levels
 * \f$\{-L,\ldots,L\}\f$, projected to L1 radius \f$L\f$, and transformed as
 * \f$x_k+\delta_k D y\f$. Zero vectors are discarded. Deduplication and the
 * maximum batch count can produce fewer candidates than requested.
 */
class LatinHypercubeMeshGPSSearchStrategy : public GPSSearchStrategy
{
  public:
  /**
   * @brief Validates active search settings and constructs its sampler.
   * @param[in] configuration Target, radius, batch cap, deduplication, and seed.
   * @throws std::invalid_argument If an active search has zero radius or zero maximum batches.
   */
  explicit LatinHypercubeMeshGPSSearchStrategy( LatinHypercubeMeshGPSSearchConfiguration configuration );

  /** @copydoc GPSSearchStrategy::ask() */
  std::vector< Eigen::VectorXd > ask( const CandidateEvaluation &current_candidate, double mesh_size,
                                      Eigen::Ref< const Eigen::MatrixXd > directions,
                                      std::size_t iteration ) override;

  /** @copydoc GPSSearchStrategy::tell() */
  void tell( const CandidateEvaluation &current_candidate, const std::vector< CandidateEvaluation > &evaluations,
             double mesh_size ) override;
  /** @brief Resets the Latin hypercube sampler. */
  void reset() override;

  private:
  /** Copied stratified mesh-search configuration. */
  LatinHypercubeMeshGPSSearchConfiguration configuration_;
  /** Unit-hypercube design generator. */
  LatinHypercubeSampler sampler_;
};

/**
 * @brief Searches affine combinations of the latest successful mesh direction.
 *
 * Before an improvement, an owned random-mesh strategy is used. Thereafter,
 * each configured multiplier \f$m\f$ is combined with every column of
 * \f$V_k=[0,I]\f$ as \f$m d_{succ}+D v\f$.
 */
class SuccessfulDirectionGPSSearchStrategy : public GPSSearchStrategy
{
  public:
  /**
   * @brief Stores multipliers and constructs the initial random search.
   * @param[in] configuration Multipliers and initial strategy settings.
   */
  explicit SuccessfulDirectionGPSSearchStrategy( SuccessfulDirectionGPSSearchConfiguration configuration );

  /** @copydoc GPSSearchStrategy::ask() */
  std::vector< Eigen::VectorXd > ask( const CandidateEvaluation &current_candidate, double mesh_size,
                                      Eigen::Ref< const Eigen::MatrixXd > directions, std::size_t iteration ) override;

  /**
   * @brief Reconstructs and stores the best strict-improvement direction.
   * @param[in] current_candidate Incumbent before the batch.
   * @param[in] evaluations Search or poll results.
   * @param[in] mesh_size Generation mesh size used in
   * \f$d_{\mathrm{succ}}=(x_{\mathrm{best}}-x_k)/\delta_k\f$.
   */
  void tell( const CandidateEvaluation &current_candidate, const std::vector< CandidateEvaluation > &evaluations,
             double mesh_size ) override;
  /** @brief Clears learned direction and basis state and resets initial search. */
  void reset() override;

  private:
  /** Copied multipliers and initial-search configuration. */
  SuccessfulDirectionGPSSearchConfiguration configuration_;
  /** Owned fallback search used until a direction succeeds. */
  RandomMeshGPSSearchStrategy initial_search_strategy_;
  /** Integer coefficient matrix \f$[0,I]\f$, sized on first ask(). */
  Eigen::MatrixXi V_k_;
  /** Latest improving direction in transformed mesh coordinates. */
  Eigen::VectorXd d_succ_;
  /** Whether d_succ_ is available. */
  bool has_successful_direction_ = false;
};

/**
 * @brief Ranks a random mesh pool with an owned local surrogate model.
 *
 * Successful search and poll evaluations accumulate in history. When fit()
 * succeeds, predicted minimization values sort the pool and the configured
 * leading fraction is returned. A failed fit returns the entire random pool.
 */
class SurrogateGPSSearchStrategy : public GPSSearchStrategy
{
  public:
  /**
   * @brief Takes ownership of the regression model and constructs the random pool generator.
   * @param[in] configuration Pool, model, and selection settings.
   * @param[in] surrogate_model Non-null model owned by the strategy.
   * @throws std::invalid_argument If the model is null or selected_fraction is outside `(0,1]`.
   */
  SurrogateGPSSearchStrategy( SurrogateGPSSearchConfiguration configuration,
                              std::unique_ptr< SurrogateModel > surrogate_model );

  /** @copydoc GPSSearchStrategy::ask() */
  std::vector< Eigen::VectorXd > ask( const CandidateEvaluation &current_candidate, double mesh_size,
                                      Eigen::Ref< const Eigen::MatrixXd > directions, std::size_t iteration ) override;

  /**
   * @brief Appends every successful trial to model history.
   * @param[in] current_candidate Incumbent supplied by GPS.
   * @param[in] evaluations Search or poll results in request order.
   * @param[in] mesh_size Mesh size used for the batch.
   */
  void tell( const CandidateEvaluation &current_candidate, const std::vector< CandidateEvaluation > &evaluations,
             double mesh_size ) override;
  /** @brief Clears history and resets both pool generator and model. */
  void reset() override;

  private:
  /** Owned random mesh pool generator. */
  RandomMeshGPSSearchStrategy random_strategy_;
  /** Copied pool, model, and fraction configuration. */
  SurrogateGPSSearchConfiguration configuration_;
  /** Owned regression model. */
  std::unique_ptr< SurrogateModel > surrogate_model_;
  /** Successful search and poll observations in arrival order. */
  std::vector< CandidateEvaluation > history_;
};

/** @} */

#endif // !OPTIMIZER_GPS_SEARCH_STRATEGY_H
