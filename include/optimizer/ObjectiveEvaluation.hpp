#ifndef OBJECTIVE_EVALUATION_H
#define OBJECTIVE_EVALUATION_H

/** @addtogroup optimizer_core_api
 * @{ */


#include <cstddef>
#include <vector>
#include "ExecutionMetrics.hpp"

/**
 * @brief Stores raw objective samples and the work used to obtain them.
 *
 * Sample values remain present when marked failed. Statistical accessors use
 * every appended value; failedSampleCount() is an independent diagnostic
 * count. Optimizer accepts an evaluation only when it has at least one sample
 * and every value is finite.
 */
class ObjectiveEvaluation
{
  public:
  /** @brief Constructs an evaluation with no samples or execution metrics. */
  ObjectiveEvaluation() = default;

  /**
   * @brief Appends one raw objective observation.
   * @param[in] value Observation in the problem's declared direction.
   * @param[in] failed Whether to increment the failed-sample counter.
   */
  void appendSample( double value, bool failed = false );
  /**
   * @brief Concatenates another aggregate into this one.
   * @param[in] other Samples, failed count, and metrics to append.
   * @post Existing samples precede @p other's samples.
   */
  void append( const ObjectiveEvaluation &other );

  /**
   * @brief Computes the incremental arithmetic mean of all samples.
   * @return Arithmetic mean, including values marked failed.
   * @throws std::logic_error If no sample has been appended.
   */
  double meanValue() const;
  /**
   * @brief Computes population standard deviation.
   * @return \f$\sqrt{\sum_i(x_i-\bar{x})^2/n}\f$ over all samples.
   * @throws std::logic_error If no sample has been appended.
   */
  double standardDeviation() const;
  /**
   * @brief Computes Bessel-corrected sample standard deviation.
   * @return \f$\sqrt{\sum_i(x_i-\bar{x})^2/(n-1)}\f$.
   * @throws std::logic_error If fewer than two samples have been appended.
   */
  double standardDeviationBesselCorrection() const;

  /** @brief Returns the number of stored raw observations. @return Sample cardinality. */
  std::size_t sampleCount() const;
  /** @brief Returns the number of observations marked failed. @return Failed-sample count. */
  std::size_t failedSampleCount() const;
  /** @brief Checks every stored value for finiteness. @return `true` for an empty aggregate or if all samples are finite. */
  bool allFinite() const;

  /**
   * @brief Accumulates work counters into this evaluation.
   * @param[in] metrics Counters to add component by component.
   */
  void appendExecutionMetrics( const ExecutionMetrics &metrics );
  /** @brief Returns accumulated work counters. @return Read-only reference owned by this evaluation. */
  const ExecutionMetrics &executionMetrics() const;

  /**
   * @brief Creates a sign-inverted copy for direction normalization.
   * @return Copy with every sample negated and all status counts and metrics retained.
   */
  ObjectiveEvaluation negated() const;

  private:
  /** Raw objective observations in append order. */
  std::vector< double > samples_;
  /** Number of appendSample() observations carrying the failed flag. */
  std::size_t failed_sample_count_ = 0;
  /** Work counters accumulated from every contributing evaluation. */
  ExecutionMetrics execution_metrics_;
};

/** @} */

#endif // !OBJECTIVE_EVALUATION_H
