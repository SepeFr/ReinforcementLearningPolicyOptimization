#include "FeedForwardNetwork.hpp"
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <iterator>
#include <stdexcept>
#include <strings.h>
#include "DenseLayer.hpp"
#include "FeedForwardNetworkConfiguration.hpp"
#include "ParametersLayout.hpp"


FeedForwardNetwork::FeedForwardNetwork( const FeedForwardNetworkConfiguration &configuration ) :
    configuration_( configuration )
{
  // Validate checked parameter arithmetic and every size_t-to-Eigen dimension conversion before allocation.
  (void) FeedForwardNetworkConfiguration::parameterCountFromConfiguration( configuration_ );

  if ( configuration_.input_size == 0 )
  {
    throw std::invalid_argument( "FeedForwardNetwork: input size must be greater than zero" );
  }
  if ( configuration_.output_size == 0 )
  {
    throw std::invalid_argument( "FeedForwardNetwork: output size must be greater than zero" );
  }
  for ( const std::size_t hidden_layer_size : configuration_.hidden_layers )
  {
    if ( hidden_layer_size == 0 )
    {
      throw std::invalid_argument( "FeedForwardNetwork: hidden layer size must be greater than zero" );
    }
  }

  layers_.reserve( configuration_.hidden_layers.size() + 1 );

  // Creating Hidden Layers
  std::size_t last_layer_output_size = configuration_.input_size;
  for ( std::size_t index = 0; index < configuration_.hidden_layers.size(); index++ )
  {
    std::size_t layer_output_size = configuration_.hidden_layers[index];

    layers_.emplace_back( last_layer_output_size, layer_output_size, configuration_.hidden_activation,
                          configuration_.use_bias );

    last_layer_output_size = layer_output_size;
  }

  layers_.emplace_back( last_layer_output_size, configuration_.output_size, configuration_.output_activation,
                        configuration_.use_bias );

  parameter_layout_ = ParameterLayout( layers_.size() );
  std::size_t total = 0;
  for ( std::size_t index = 0; index < layers_.size(); index++ )
  {

    LayerParameterLayout layer_parameter_layout;
    layer_parameter_layout.weights_count = layers_[index].weightCount();
    layer_parameter_layout.biases_count = layers_[index].biasCount();

    layer_parameter_layout.weights_offset = total;
    layer_parameter_layout.biases_offset = total + layer_parameter_layout.weights_count;

    total += layers_[index].parameterCount();

    parameter_layout_.setLayer( index, layer_parameter_layout );
  }
}


Eigen::VectorXd FeedForwardNetwork::forward( Eigen::Ref< const Eigen::VectorXd > input ) const
{
  Eigen::VectorXd last_layer_output = input;
  for ( const DenseLayer &layer : layers_ )
  {
    Eigen::VectorXd layer_output = layer.forward( last_layer_output );
    last_layer_output = layer_output;
  }
  return last_layer_output;
}


std::size_t FeedForwardNetwork::parameterCount() const { return parameter_layout_.totalParameterCount(); }


void FeedForwardNetwork::setParameters( Eigen::Ref< const Eigen::VectorXd > parameters )
{
  Eigen::Index number_of_parameters = static_cast< Eigen::Index >( parameterCount() );
  if ( parameters.size() != number_of_parameters )
  {
    throw std::invalid_argument( "FeedForwardNetwork: invalid parameter count" );
  }
  std::size_t total = 0;

  for ( DenseLayer &layer : layers_ )
  {
    layer.setParameters( parameters.segment( total, layer.parameterCount() ) );
    total += layer.parameterCount();
  }
}


Eigen::VectorXd FeedForwardNetwork::parameters() const
{
  Eigen::VectorXd output;
  for ( const DenseLayer &layer : layers_ )
  {
    layer.appendParameters( output );
  }
  return output;
}


Eigen::VectorXd FeedForwardNetwork::weightsParameters() const
{
  Eigen::VectorXd output;
  for ( const DenseLayer &layer : layers_ )
  {
    layer.appendWeights( output );
  }
  return output;
}


Eigen::VectorXd FeedForwardNetwork::biasesParameters() const
{
  Eigen::VectorXd output;
  for ( const DenseLayer &layer : layers_ )
  {
    layer.appendBiases( output );
  }
  return output;
}
