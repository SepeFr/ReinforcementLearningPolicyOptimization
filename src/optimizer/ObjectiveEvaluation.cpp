#include "ObjectiveEvaluation.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>


void ObjectiveEvaluation::appendSample( double value, bool failed )
{
  if ( failed )
  {
    failed_sample_count_++;
  }
  samples_.push_back( value );
}

void ObjectiveEvaluation::append( const ObjectiveEvaluation &other )
{
  // Preserve every raw observation so cumulative mean, variance and sample
  // count remain correct when a stochastic method resamples a candidate.
  samples_.reserve( samples_.size() + other.samples_.size() );
  samples_.insert( samples_.end(), other.samples_.begin(), other.samples_.end() );
  failed_sample_count_ += other.failed_sample_count_;
  execution_metrics_.append( other.execution_metrics_ );
}


double ObjectiveEvaluation::meanValue() const
{
  if ( samples_.empty() )
  {
    throw std::logic_error( "ObjectiveEvaluation: cannot calculate the mean without samples" );
  }

  double mean = 0.0;
  for ( std::size_t index = 0; index < samples_.size(); ++index )
  {
    const double sample = samples_[index];
    mean += ( sample - mean ) / static_cast< double >( index + 1 );
  }
  return mean;
}


double ObjectiveEvaluation::standardDeviation() const
{
  if ( samples_.empty() )
  {
    throw std::logic_error( "ObjectiveEvaluation: cannot calculate standard deviation without samples" );
  }

  double mean = meanValue();
  double variance = 0.0;

  for ( const double sample : samples_ )
  {
    const double difference = sample - mean;
    variance += difference * difference;
  }

  return std::sqrt( variance / static_cast< double >( samples_.size() ) );
}

double ObjectiveEvaluation::standardDeviationBesselCorrection() const
{
  if ( samples_.size() < 2 )
  {
    throw std::logic_error( "ObjectiveEvaluation: Bessel-corrected standard deviation requires at least two samples" );
  }

  const double mean = meanValue();
  double squared_deviation_sum = 0.0;

  for ( const double sample : samples_ )
  {
    const double difference = sample - mean;
    squared_deviation_sum += difference * difference;
  }

  return std::sqrt( squared_deviation_sum / static_cast< double >( samples_.size() - 1 ) );
}


std::size_t ObjectiveEvaluation::sampleCount() const { return samples_.size(); }


std::size_t ObjectiveEvaluation::failedSampleCount() const { return failed_sample_count_; }


bool ObjectiveEvaluation::allFinite() const
{
  for ( const double sample : samples_ )
  {
    if ( !std::isfinite( sample ) )
    {
      return false;
    }
  }
  return true;
}


void ObjectiveEvaluation::appendExecutionMetrics( const ExecutionMetrics &metrics )
{
  execution_metrics_.append( metrics );
}


const ExecutionMetrics &ObjectiveEvaluation::executionMetrics() const { return execution_metrics_; }


ObjectiveEvaluation ObjectiveEvaluation::negated() const
{
  ObjectiveEvaluation result = *this;
  for ( double &sample : result.samples_ )
  {
    sample = -sample;
  }
  return result;
}
