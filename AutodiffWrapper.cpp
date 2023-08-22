#include "AutodiffWrapper.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
using namespace autodiff;

AutodiffWrapper::AutodiffWrapper()
{

    sigmaFn s1 = [](VectorXdual u, VectorXdual q, dual x, double t)
    {
        return cos(u.sum()) + x;
    };
    sigmaFn s2 = [](VectorXdual u, VectorXdual q, dual x, double t)
    {
        return 5 * q(1);
    };
    sigmaFn s3 = [](VectorXdual u, VectorXdual q, dual x, double t)
    {
        return cos(x);
    };

    sigmaVec.push_back(s1);
    sigmaVec.push_back(s2);
    sigmaVec.push_back(s3);
}

Value AutodiffWrapper::LowerBoundary(Index i, Time t) const { return 0; }

Value AutodiffWrapper::UpperBoundary(Index i, Time t) const { return 0; }

bool AutodiffWrapper::isLowerBoundaryDirichlet(Index i) const { return true; }

bool AutodiffWrapper::isUpperBoundaryDirichlet(Index i) const { return true; }

// The same for the flux and source functions -- the vectors have length nVars

Value AutodiffWrapper::SigmaFn(Index i, const Values &u, const Values &q, Position x, Time t)
{
    vec udual(u);
    vec qdual(q);
    dual pos = x;
    double sigma = sigmaVec[i](udual, qdual, pos, t).val;
    return sigma;
}
Value AutodiffWrapper::Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t) { return 0; }

// We need derivatives of the flux functions
Value AutodiffWrapper::dSigmaFn_du(Index i, Index j, const Values &u, const Values &q, Position x, Time t)
{

    vec udual(u);
    vec qdual(q);

    auto dsdu = gradient(sigmaVec[i], wrt(udual(j)), at(udual, qdual, x, t));
    return (Value)dsdu(0);
}
Value AutodiffWrapper::dSigmaFn_dq(Index i, Index j, const Values &u, const Values &q, Position x, Time t)
{
    vec udual(u);
    vec qdual(q);

    auto dsdq = gradient(sigmaVec[i], wrt(qdual(j)), at(udual, qdual, x, t));
    return (Value)dsdq(0);
}

// and for the sources
Value AutodiffWrapper::dSources_du(Index i, Index j, const Values &u, const Values &q, Position x, Time t) { return 0; }
Value AutodiffWrapper::dSources_dq(Index i, Index j, const Values &u, const Values &q, Position x, Time t) { return 0; }
Value AutodiffWrapper::dSources_dsigma(Index i, Index j, const Values &u, const Values &q, Position x, Time t) { return 0; }

// and initial conditions for u & q
Value AutodiffWrapper::InitialValue(Index i, Position x) const { return 0; }
Value AutodiffWrapper::InitialDerivative(Index i, Position x) const { return 0; }
