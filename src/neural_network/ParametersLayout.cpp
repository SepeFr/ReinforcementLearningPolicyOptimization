#include "ParametersLayout.hpp"
#include <cstddef>
#include <stdexcept>


ParameterLayout::ParameterLayout( std::size_t number_of_layers ) : layout_parameters_( number_of_layers ) {}
std::size_t ParameterLayout::totalParameterCount() const
{
  std::size_t total_count = 0;
  for ( const LayerParameterLayout &layer : layout_parameters_ )
  {
    total_count += layer.weights_count + layer.biases_count;
  }
  return total_count;
}


const LayerParameterLayout &ParameterLayout::layer( std::size_t index ) const
{
  return layout_parameters_.at( index );
}

void ParameterLayout::setLayer( std::size_t index, const LayerParameterLayout &layout )
{
  if ( index >= layout_parameters_.size() )
  {
    throw std::out_of_range( "ParameterLayout: layer index out of range" );
  }
  layout_parameters_.at( index ) = layout;
}
