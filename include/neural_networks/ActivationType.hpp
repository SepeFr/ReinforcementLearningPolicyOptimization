#ifndef ACTIVATION_TYPE_H
#define ACTIVATION_TYPE_H

/** @addtogroup neural_network_api
 * @{ */

/** @file ActivationType.hpp @brief Element-wise neural-network activation identifiers. */

/** @brief Element-wise activation applied after a dense affine transform. */
enum class ActivationType
{
  Linear,  ///< Identity, \f$ f(x)=x \f$.
  Tanh,    ///< Hyperbolic tangent, \f$ f(x)=\tanh(x) \f$.
  Sigmoid, ///< Logistic function, \f$ f(x)=1/(1+e^{-x}) \f$.
  ReLu     ///< Rectified linear unit, \f$ f(x)=\max(0,x) \f$.
};

/** @} */

#endif // ! ACTIVATION_TYPE_H
