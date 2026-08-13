#ifndef LATIN_HYPERCUBE_SAMPLER_H
#define LATIN_HYPERCUBE_SAMPLER_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <cstddef>
#include <cstdint>
#include <eigen3/Eigen/Core>
#include <random>

/**
 * @brief Generates reproducible Latin hypercube designs on the unit cube.
 *
 * Rows are sampled points and columns are dimensions. In each column every one
 * of the point_count equal strata of \f$[0,1)\f$ is used exactly once.
 */
class LatinHypercubeSampler
{
  public:
  /** @brief Constructs a sampler with a deterministic seed. @param[in] seed Seed restored by reset(). */
  explicit LatinHypercubeSampler( std::uint64_t seed ) : seed_( seed ), generator_( seed_ ) {}

  /**
   * @brief Generates one stratified unit-cube design.
   * @param[in] point_count Number of design rows.
   * @param[in] dimension Number of coordinate columns.
   * @return `point_count` by `dimension` matrix with values in \f$[0,1)\f$.
   * @throws std::invalid_argument If either argument is zero.
   */
  Eigen::MatrixXd generate( std::size_t point_count, std::size_t dimension );
  /** @brief Restores the engine to its construction seed. */
  void reset() { generator_.seed( seed_ ); }

  private:
  /** Seed retained for deterministic reset. */
  std::uint64_t seed_;
  /** Random engine used for stratum permutations and within-stratum offsets. */
  std::mt19937_64 generator_;
};

/** @} */

#endif // !LATIN_HYPERCUBE_SAMPLER_H
