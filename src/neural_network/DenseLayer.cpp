#include "DenseLayer.hpp"
#include <cassert>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <iterator>
#include <stdexcept>
#include <utility>
#include "ActivationFunction.hpp"

namespace
{
  Eigen::Index toEigenIndex( std::size_t value )
  {
    if ( !std::in_range< Eigen::Index >( value ) )
    {
      throw std::overflow_error( "DenseLayer: layer size is not representable as Eigen::Index" );
    }
    return static_cast< Eigen::Index >( value );
  }
} // namespace

DenseLayer::DenseLayer( std::size_t input_size, std::size_t output_size, ActivationType activation_type,
                        bool use_bias ) :
    weights_( toEigenIndex( output_size ), toEigenIndex( input_size ) ),
    biases_( use_bias ? toEigenIndex( output_size ) : 0 ), activation_( activation_type ),
    use_bias_( use_bias )
{
  weights_.setZero();
  biases_.setZero();
}


std::size_t DenseLayer::parameterCount() const { return weights_.size() + biases_.size(); }
std::size_t DenseLayer::weightCount() const { return weights_.size(); }
std::size_t DenseLayer::biasCount() const { return biases_.size(); }

void DenseLayer::setParameters( Eigen::Ref< const Eigen::VectorXd > parameters )
{
  const Eigen::Index weightsCount = weights_.size();
  const Eigen::Index biasesCount = biases_.size();
  const Eigen::Index totalCount = weightsCount + biasesCount;

  if ( parameters.size() != totalCount )
  {
    throw std::invalid_argument( "DenseLayer: invalid parameter count" );
  }

  Eigen::Map< const Eigen::MatrixXd > mappedWeights( parameters.data(), weights_.rows(), weights_.cols() );
  weights_ = mappedWeights;

  if ( use_bias_ )
  {
    biases_ = parameters.segment( weightsCount, biasesCount );
  }
}


void DenseLayer::appendParameters( Eigen::VectorXd &destination ) const
{
  const Eigen::Index weightsCount = weights_.size();
  const Eigen::Index biasesCount = biases_.size();
  const Eigen::Index totalCount = weightsCount + biasesCount;
  const Eigen::Index oldSize = destination.size();

  destination.conservativeResize( oldSize + totalCount );

  destination.segment( oldSize, weightsCount ) = Eigen::Map< const Eigen::VectorXd >( weights_.data(), weightsCount );
  if ( use_bias_ )
  {
    destination.segment( oldSize + weightsCount, biasesCount ) =
      Eigen::Map< const Eigen::VectorXd >( biases_.data(), biasesCount );
  }
}


Eigen::VectorXd DenseLayer::forward( Eigen::Ref< const Eigen::VectorXd > input ) const
{

  if ( input.size() != weights_.cols() )
  {
    throw std::invalid_argument( "DenseLayer: input vector size differs from weights columns" );
  }
  Eigen::VectorXd output = weights_ * input;
  if ( use_bias_ )
  {
    output += biases_;
  }
  return ActivationFunction::apply( activation_, output );
}


void DenseLayer::appendWeights( Eigen::VectorXd &destination ) const
{
  const Eigen::Index weightsCount = weights_.size();
  const Eigen::Index oldSize = destination.size();

  destination.conservativeResize( oldSize + weightsCount );

  destination.segment( oldSize, weightsCount ) = Eigen::Map< const Eigen::VectorXd >( weights_.data(), weightsCount );
}


void DenseLayer::appendBiases( Eigen::VectorXd &destination ) const
{
  const Eigen::Index biasesCount = biases_.size();
  const Eigen::Index oldSize = destination.size();

  destination.conservativeResize( oldSize + biasesCount );

  if ( use_bias_ )
  {
    destination.segment( oldSize, biasesCount ) = Eigen::Map< const Eigen::VectorXd >( biases_.data(), biasesCount );
  }
}
