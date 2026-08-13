#include "BlackBoxProblem.hpp"
#include <eigen3/Eigen/Core>
#include <stdexcept>
#include <utility>

namespace
{
  void requireEigenIndex( std::size_t value )
  {
    if ( !std::in_range< Eigen::Index >( value ) )
    {
      throw std::overflow_error( "BlackBoxProblem: parameter count is not representable as Eigen::Index" );
    }
  }

  std::size_t toSize( Eigen::Index value )
  {
    if ( !std::in_range< std::size_t >( value ) )
    {
      throw std::overflow_error( "BlackBoxProblem: Eigen dimension is not representable as size_t" );
    }
    return static_cast< std::size_t >( value );
  }
} // namespace

BlackBoxProblem::BlackBoxProblem( std::size_t parameters_count, OptimizationDirection direction ) :
    parameters_count_( parameters_count ), direction_( direction )
{
  requireEigenIndex( parameters_count_ );
}

BlackBoxProblem::BlackBoxProblem( Eigen::Ref< const Eigen::VectorXd > lower_bound,
                                  Eigen::Ref< const Eigen::VectorXd > upper_bound, OptimizationDirection direction ) :
    parameters_count_( toSize( lower_bound.size() ) ), lower_bound_( lower_bound ), upper_bound_( upper_bound ),
    direction_( direction )
{
  if ( lower_bound_->size() != upper_bound_->size() )
  {
    throw std::invalid_argument( "BlackBoxProblem: lower and upper bounds must have the same size" );
  }
  if ( !( lower_bound_->array() <= upper_bound_->array() ).all() )
  {
    throw std::invalid_argument( "BlackBoxProblem: lower bound cannot exceed upper bound" );
  }
}


BlackBoxProblem::BlackBoxProblem( Eigen::Ref< const Eigen::VectorXd > bound, bool is_lower_bound,
                                  OptimizationDirection direction ) :
    parameters_count_( toSize( bound.size() ) ), direction_( direction )
{
  if ( is_lower_bound )
  {
    lower_bound_ = bound;
  }
  else
  {
    upper_bound_ = bound;
  }
}


bool BlackBoxProblem::withinBounds( Eigen::Ref< const Eigen::VectorXd > input ) const
{
  if ( input.size() != static_cast< Eigen::Index >( parameters_count_ ) )
  {
    return false;
  }

  if ( lower_bound_.has_value() )
  {
    if ( lower_bound_->size() != input.size() || !( input.array() >= lower_bound_->array() ).all() )
    {
      return false;
    }
  }

  if ( upper_bound_.has_value() )
  {
    if ( upper_bound_->size() != input.size() || !( input.array() <= upper_bound_->array() ).all() )
    {
      return false;
    }
  }

  return true;
}


std::size_t BlackBoxProblem::parametersCount() const { return parameters_count_; }
OptimizationDirection BlackBoxProblem::direction() const { return direction_; }
const std::optional< Eigen::VectorXd > &BlackBoxProblem::lowerBound() const { return lower_bound_; }
const std::optional< Eigen::VectorXd > &BlackBoxProblem::upperBound() const { return upper_bound_; }
