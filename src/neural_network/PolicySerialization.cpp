#include "PolicySerialization.hpp"
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "ActivationType.hpp"
#include "FeedForwardNetworkConfiguration.hpp"
#include "PolicyConfiguration.hpp"
#include "PolicyFactory.hpp"

namespace
{
  constexpr std::size_t format_version = 1;

  std::string policyTypeName( PolicyType type )
  {
    switch ( type )
    {
      case PolicyType::FeedForwardEigen:
        return "FeedForwardEigen";
    }

    throw std::invalid_argument( "PolicySerialization: unsupported policy type" );
  }

  PolicyType parsePolicyType( const std::string &value )
  {
    if ( value == "FeedForwardEigen" )
    {
      return PolicyType::FeedForwardEigen;
    }

    throw std::runtime_error( "PolicySerialization: unsupported policy type in file" );
  }

  std::string activationTypeName( ActivationType type )
  {
    switch ( type )
    {
      case ActivationType::Linear:
        return "Linear";
      case ActivationType::Tanh:
        return "Tanh";
      case ActivationType::Sigmoid:
        return "Sigmoid";
      case ActivationType::ReLu:
        return "ReLu";
    }

    throw std::invalid_argument( "PolicySerialization: unsupported activation type" );
  }

  ActivationType parseActivationType( const std::string &value )
  {
    if ( value == "Linear" )
    {
      return ActivationType::Linear;
    }
    if ( value == "Tanh" )
    {
      return ActivationType::Tanh;
    }
    if ( value == "Sigmoid" )
    {
      return ActivationType::Sigmoid;
    }
    if ( value == "ReLu" )
    {
      return ActivationType::ReLu;
    }

    throw std::runtime_error( "PolicySerialization: unsupported activation type in file" );
  }

  std::string regularizationTypeName( RegularizationType type )
  {
    switch ( type )
    {
      case RegularizationType::L2_Regularization:
        return "L2_Regularization";
      case RegularizationType::L1_Regularization:
        return "L1_Regularization";
    }

    throw std::invalid_argument( "PolicySerialization: unsupported regularization type" );
  }

  RegularizationType parseRegularizationType( const std::string &value )
  {
    if ( value == "L2_Regularization" )
    {
      return RegularizationType::L2_Regularization;
    }
    if ( value == "L1_Regularization" )
    {
      return RegularizationType::L1_Regularization;
    }

    throw std::runtime_error( "PolicySerialization: unsupported regularization type in file" );
  }

  std::vector< std::string > splitRow( const std::string &line )
  {
    std::vector< std::string > fields;
    std::size_t begin = 0;

    while ( true )
    {
      const std::size_t separator = line.find( ',', begin );
      fields.push_back( line.substr( begin, separator - begin ) );

      if ( separator == std::string::npos )
      {
        return fields;
      }

      begin = separator + 1;
    }
  }

  std::vector< std::string > readRow( std::ifstream &input, const std::string &expected_key )
  {
    std::string line;
    if ( !std::getline( input, line ) )
    {
      throw std::runtime_error( "PolicySerialization: missing row " + expected_key );
    }

    if ( !line.empty() && line.back() == '\r' )
    {
      line.pop_back();
    }

    std::vector< std::string > fields = splitRow( line );
    if ( fields.empty() || fields.front() != expected_key )
    {
      throw std::runtime_error( "PolicySerialization: expected row " + expected_key );
    }

    fields.erase( fields.begin() );
    return fields;
  }

  std::string readSingleValue( std::ifstream &input, const std::string &key )
  {
    std::vector< std::string > values = readRow( input, key );
    if ( values.size() != 1 )
    {
      throw std::runtime_error( "PolicySerialization: row " + key + " must contain one value" );
    }

    return std::move( values.front() );
  }

  std::size_t parseSize( const std::string &value, const std::string &field_name )
  {
    try
    {
      if ( value.empty() || value.front() == '-' )
      {
        throw std::runtime_error( "invalid value" );
      }

      std::size_t parsed_characters = 0;
      const unsigned long long parsed_value = std::stoull( value, &parsed_characters );

      if ( parsed_characters != value.size() || parsed_value > std::numeric_limits< std::size_t >::max() )
      {
        throw std::runtime_error( "invalid value" );
      }

      return static_cast< std::size_t >( parsed_value );
    }
    catch ( const std::exception & )
    {
      throw std::runtime_error( "PolicySerialization: invalid " + field_name );
    }
  }

  double parseFiniteDouble( const std::string &value, const std::string &field_name )
  {
    try
    {
      std::size_t parsed_characters = 0;
      const double parsed_value = std::stod( value, &parsed_characters );

      if ( parsed_characters != value.size() || !std::isfinite( parsed_value ) )
      {
        throw std::runtime_error( "invalid value" );
      }

      return parsed_value;
    }
    catch ( const std::exception & )
    {
      throw std::runtime_error( "PolicySerialization: invalid " + field_name );
    }
  }

  bool parseBool( const std::string &value )
  {
    if ( value == "0" )
    {
      return false;
    }
    if ( value == "1" )
    {
      return true;
    }

    throw std::runtime_error( "PolicySerialization: invalid use_bias" );
  }

  void validatePolicy( const PolicyConfiguration &configuration, Eigen::Ref< const Eigen::VectorXd > parameters )
  {
    if ( !std::isfinite( configuration.regularization_coefficient ) || !parameters.allFinite() )
    {
      throw std::invalid_argument( "PolicySerialization: policy contains non-finite values" );
    }

    static_cast< void >( PolicyFactory::create( configuration, parameters ) );
  }
} // namespace

void PolicySerialization::serialize( const std::filesystem::path &path, const PolicyConfiguration &configuration,
                                     Eigen::Ref< const Eigen::VectorXd > parameters )
{
  validatePolicy( configuration, parameters );

  std::ofstream output( path );
  if ( !output )
  {
    throw std::runtime_error( "PolicySerialization: cannot open output file" );
  }

  output << std::setprecision( std::numeric_limits< double >::max_digits10 );
  output << "format_version," << format_version << '\n';
  output << "policy_type," << policyTypeName( configuration.type ) << '\n';
  output << "input_size," << configuration.network.input_size << '\n';
  output << "hidden_layers";
  for ( const std::size_t layer_size : configuration.network.hidden_layers )
  {
    output << ',' << layer_size;
  }
  output << '\n';
  output << "output_size," << configuration.network.output_size << '\n';
  output << "hidden_activation," << activationTypeName( configuration.network.hidden_activation ) << '\n';
  output << "output_activation," << activationTypeName( configuration.network.output_activation ) << '\n';
  output << "use_bias," << ( configuration.network.use_bias ? 1 : 0 ) << '\n';
  output << "regularization_type," << regularizationTypeName( configuration.regularization_strategy ) << '\n';
  output << "regularization_coefficient," << configuration.regularization_coefficient << '\n';
  output << "parameters";
  for ( Eigen::Index index = 0; index < parameters.size(); index++ )
  {
    output << ',' << parameters[index];
  }
  output << '\n';

  if ( !output )
  {
    throw std::runtime_error( "PolicySerialization: failed to write policy" );
  }
}

PolicySerialization::DeserializedPolicy PolicySerialization::deserialize( const std::filesystem::path &path )
{
  std::ifstream input( path );
  if ( !input )
  {
    throw std::runtime_error( "PolicySerialization: cannot open input file" );
  }

  const std::size_t stored_version = parseSize( readSingleValue( input, "format_version" ), "format_version" );
  if ( stored_version != format_version )
  {
    throw std::runtime_error( "PolicySerialization: unsupported format version" );
  }

  const PolicyType policy_type = parsePolicyType( readSingleValue( input, "policy_type" ) );

  FeedForwardNetworkConfiguration network_configuration;
  network_configuration.input_size = parseSize( readSingleValue( input, "input_size" ), "input_size" );

  const std::vector< std::string > hidden_layer_values = readRow( input, "hidden_layers" );
  network_configuration.hidden_layers.reserve( hidden_layer_values.size() );
  for ( const std::string &value : hidden_layer_values )
  {
    network_configuration.hidden_layers.push_back( parseSize( value, "hidden layer size" ) );
  }

  network_configuration.output_size = parseSize( readSingleValue( input, "output_size" ), "output_size" );
  network_configuration.hidden_activation =
    parseActivationType( readSingleValue( input, "hidden_activation" ) );
  network_configuration.output_activation =
    parseActivationType( readSingleValue( input, "output_activation" ) );
  network_configuration.use_bias = parseBool( readSingleValue( input, "use_bias" ) );

  const RegularizationType regularization_type =
    parseRegularizationType( readSingleValue( input, "regularization_type" ) );
  const double regularization_coefficient =
    parseFiniteDouble( readSingleValue( input, "regularization_coefficient" ), "regularization_coefficient" );

  const std::vector< std::string > parameter_values = readRow( input, "parameters" );
  Eigen::VectorXd parameters( static_cast< Eigen::Index >( parameter_values.size() ) );
  for ( std::size_t index = 0; index < parameter_values.size(); index++ )
  {
    parameters[static_cast< Eigen::Index >( index )] = parseFiniteDouble( parameter_values[index], "parameter" );
  }

  std::string extra_line;
  while ( std::getline( input, extra_line ) )
  {
    if ( !extra_line.empty() && extra_line != "\r" )
    {
      throw std::runtime_error( "PolicySerialization: unexpected data after parameters" );
    }
  }

  PolicyConfiguration configuration{ policy_type, std::move( network_configuration ), regularization_type,
                                     regularization_coefficient };
  validatePolicy( configuration, parameters );

  return DeserializedPolicy{ std::move( configuration ), std::move( parameters ) };
}
