#ifndef NEIGHBORHOOD_STRATEGY_H
#define NEIGHBORHOOD_STRATEGY_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <random>
#include <vector>
#include "LatinHypercubeSampler.hpp"
#include "NeighborhoodConfiguration.hpp"

/** @brief Interface for generating local candidate perturbations. */
class NeighborhoodStrategy
{
  public:
  /** @brief Enables destruction through the strategy interface. */
  virtual ~NeighborhoodStrategy() = default;
  /** @brief Restores deterministic generator state before a new run. */
  virtual void reset() {}

  /**
   * @brief Generates one candidate around a current point.
   * @param[in] current_parameters Center coordinates.
   * @return Owning perturbed vector with matching cardinality.
   */
  virtual Eigen::VectorXd generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters ) = 0;

  /**
   * @brief Generates an ordered batch around one current point.
   *
   * The default implementation calls generateNeighbor() independently for
   * each entry. Batch-aware strategies override this function.
   *
   * @param[in] current_parameters Center coordinates shared by the batch.
   * @param[in] number_of_neighbors Requested batch cardinality.
   * @return Exactly @p number_of_neighbors vectors in generation order.
   */
  virtual std::vector< Eigen::VectorXd >
  generateNeighbors( Eigen::Ref< const Eigen::VectorXd > current_parameters, std::size_t number_of_neighbors )
  {
    std::vector< Eigen::VectorXd > neighbors;
    neighbors.reserve( number_of_neighbors );

    for ( std::size_t index = 0; index < number_of_neighbors; ++index )
    {
      neighbors.push_back( generateNeighbor( current_parameters ) );
    }

    return neighbors;
  }
};

/** @brief Adds independent Gaussian displacements to every coordinate. */
class GaussianNeighborhood : public NeighborhoodStrategy
{
  public:
  /**
   * @brief Stores distribution parameters and seeds the generator.
   * @param[in] perturbation_mean Mean coordinate displacement.
   * @param[in] standard_deviation Positive displacement standard deviation.
   * @param[in] seed Seed restored by reset().
   * @pre @p standard_deviation is positive.
   */
  GaussianNeighborhood( double perturbation_mean = 0.0, double standard_deviation = 1.0, std::size_t seed = 0 ) :
      perturbation_mean_( perturbation_mean ), standard_deviation_( standard_deviation ),
      seed_( static_cast< std::mt19937::result_type >( seed ) ), generator_( seed_ )
  {
  }

  /** @copydoc NeighborhoodStrategy::generateNeighbor() */
  Eigen::VectorXd generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters ) override
  {
    Eigen::VectorXd neighbor = current_parameters;
    std::normal_distribution< double > perturbation( perturbation_mean_, standard_deviation_ );

    for ( Eigen::Index index = 0; index < neighbor.size(); index++ )
    {
      neighbor[index] += perturbation( generator_ );
    }

    return neighbor;
  }

  /** @brief Restores the generator to its construction seed. */
  void reset() override { generator_.seed( seed_ ); }

  private:
  /** Mean of each independent displacement. */
  double perturbation_mean_;
  /** Standard deviation of each independent displacement. */
  double standard_deviation_;
  /** Seed converted to the generator's result type. */
  std::mt19937::result_type seed_;
  /** Random engine advanced by generated coordinates. */
  std::mt19937 generator_;
};

/** @brief Adds independent uniform displacements to every coordinate. */
class UniformNeighborhood : public NeighborhoodStrategy
{
  public:
  /**
   * @brief Stores the interval and seeds the generator.
   * @param[in] lower_bound Lower displacement endpoint.
   * @param[in] upper_bound Upper displacement endpoint.
   * @param[in] seed Seed restored by reset().
   * @pre @p lower_bound is at most @p upper_bound.
   */
  UniformNeighborhood( double lower_bound, double upper_bound, std::size_t seed ) :
      lower_bound_( lower_bound ), upper_bound_( upper_bound ),
      seed_( static_cast< std::mt19937::result_type >( seed ) ), generator_( seed_ )
  {
  }

  /** @copydoc NeighborhoodStrategy::generateNeighbor() */
  Eigen::VectorXd generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters ) override
  {
    Eigen::VectorXd neighbor = current_parameters;
    std::uniform_real_distribution< double > perturbation( lower_bound_, upper_bound_ );

    for ( Eigen::Index index = 0; index < neighbor.size(); index++ )
    {
      neighbor[index] += perturbation( generator_ );
    }

    return neighbor;
  }

  /** @brief Restores the generator to its construction seed. */
  void reset() override { generator_.seed( seed_ ); }

  private:
  /** Lower displacement endpoint. */
  double lower_bound_;
  /** Upper displacement endpoint. */
  double upper_bound_;
  /** Seed converted to the generator's result type. */
  std::mt19937::result_type seed_;
  /** Random engine advanced by generated coordinates. */
  std::mt19937 generator_;
};

/**
 * @brief Perturbs a random number of coordinates sampled with replacement.
 *
 * The number of draws is uniform from one through the vector dimension. A
 * coordinate selected multiple times receives multiple independent additions.
 */
class CoordinateNeighborhood : public NeighborhoodStrategy
{
  public:
  /**
   * @brief Stores the displacement interval and seeds the generator.
   * @param[in] lower_bound Lower displacement endpoint.
   * @param[in] upper_bound Upper displacement endpoint.
   * @param[in] seed Seed restored by reset().
   * @pre @p lower_bound is at most @p upper_bound.
   */
  CoordinateNeighborhood( double lower_bound, double upper_bound, std::size_t seed ) :
      lower_bound_( lower_bound ), upper_bound_( upper_bound ),
      seed_( static_cast< std::mt19937::result_type >( seed ) ), generator_( seed_ )
  {
  }

  /**
   * @copybrief NeighborhoodStrategy::generateNeighbor()
   * @param[in] current_parameters Center coordinates; an empty vector is returned unchanged.
   * @return Perturbed vector with matching cardinality.
   */
  Eigen::VectorXd generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters ) override
  {
    Eigen::VectorXd neighbor = current_parameters;
    if ( neighbor.size() == 0 )
    {
      return neighbor;
    }

    std::uniform_int_distribution< Eigen::Index > coordinate_count_distribution( 1, neighbor.size() );
    std::uniform_int_distribution< Eigen::Index > coordinate_distribution( 0, neighbor.size() - 1 );
    std::uniform_real_distribution< double > perturbation( lower_bound_, upper_bound_ );

    const Eigen::Index coordinate_count = coordinate_count_distribution( generator_ );
    for ( Eigen::Index count = 0; count < coordinate_count; count++ )
    {
      // Replacement is part of the distribution: a coordinate may receive several displacements.
      const Eigen::Index index = coordinate_distribution( generator_ );
      neighbor[index] += perturbation( generator_ );
    }

    return neighbor;
  }

  /** @brief Restores the generator to its construction seed. */
  void reset() override { generator_.seed( seed_ ); }

  private:
  /** Lower displacement endpoint. */
  double lower_bound_;
  /** Upper displacement endpoint. */
  double upper_bound_;
  /** Seed converted to the generator's result type. */
  std::mt19937::result_type seed_;
  /** Random engine advanced by coordinate and displacement draws. */
  std::mt19937 generator_;
};

/**
 * @brief Maps one Latin hypercube design to uniform coordinate displacements.
 *
 * A batch of size \f$n\f$ stratifies each coordinate interval into \f$n\f$
 * strata, giving the batch-level distribution meaning.
 */
class LatinHypercubeUniformNeighborhood : public NeighborhoodStrategy
{
  public:
  /**
   * @brief Stores the interval and constructs a deterministic sampler.
   * @param[in] configuration Uniform displacement endpoints.
   * @param[in] seed Latin hypercube sampler seed.
   * @throws std::invalid_argument If lower_bound exceeds upper_bound.
   */
  LatinHypercubeUniformNeighborhood( LatinHypercubeUniformNeighborhoodConfiguration configuration,
                                     std::size_t seed );

  /**
   * @copybrief NeighborhoodStrategy::generateNeighbor()
   * @param[in] current_parameters Nonempty center coordinates.
   * @return One uniformly displaced candidate.
   * @throws std::invalid_argument If the center is empty.
   */
  Eigen::VectorXd generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters ) override;

  /**
   * @brief Generates a jointly stratified uniform batch.
   * @param[in] current_parameters Center coordinates.
   * @param[in] number_of_neighbors Batch size; zero returns an empty batch.
   * @return Candidates in Latin-hypercube row order.
   * @throws std::invalid_argument If the requested batch is nonempty and the center is empty.
   */
  std::vector< Eigen::VectorXd >
  generateNeighbors( Eigen::Ref< const Eigen::VectorXd > current_parameters,
                     std::size_t number_of_neighbors ) override;
  /** @brief Restores the Latin hypercube sampler seed. */
  void reset() override;

  private:
  /** Copied displacement endpoints. */
  LatinHypercubeUniformNeighborhoodConfiguration configuration_;
  /** Unit-cube design generator owned by the strategy. */
  LatinHypercubeSampler sampler_;
};

/** @brief Maps a Latin hypercube design through a Gaussian quantile function. */
class LatinHypercubeGaussianNeighborhood : public NeighborhoodStrategy
{
  public:
  /**
   * @brief Stores Gaussian parameters and constructs a deterministic sampler.
   * @param[in] configuration Mean and finite positive standard deviation.
   * @param[in] seed Latin hypercube sampler seed.
   * @throws std::invalid_argument If standard_deviation is non-finite or non-positive.
   */
  LatinHypercubeGaussianNeighborhood( LatinHypercubeGaussianNeighborhoodConfiguration configuration,
                                      std::size_t seed );

  /**
   * @copybrief NeighborhoodStrategy::generateNeighbor()
   * @param[in] current_parameters Nonempty center coordinates.
   * @return One normally displaced candidate.
   * @throws std::invalid_argument If the center is empty.
   */
  Eigen::VectorXd generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters ) override;

  /**
   * @brief Generates a jointly stratified Gaussian batch.
   * @param[in] current_parameters Center coordinates.
   * @param[in] number_of_neighbors Batch size; zero returns an empty batch.
   * @return Candidates in Latin-hypercube row order.
   * @throws std::invalid_argument If the requested batch is nonempty and the center is empty.
   */
  std::vector< Eigen::VectorXd >
  generateNeighbors( Eigen::Ref< const Eigen::VectorXd > current_parameters,
                     std::size_t number_of_neighbors ) override;
  /** @brief Restores the Latin hypercube sampler seed. */
  void reset() override;

  private:
  /** Copied Gaussian displacement parameters. */
  LatinHypercubeGaussianNeighborhoodConfiguration configuration_;
  /** Unit-cube design generator owned by the strategy. */
  LatinHypercubeSampler sampler_;
};

/** @} */

#endif // !NEIGHBORHOOD_STRATEGY_H
