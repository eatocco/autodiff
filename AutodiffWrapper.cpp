#include "AutodiffWrapper.hpp"
#include <autodiff/forward/dual.hpp>
#include <eigen3/Eigen/Core>
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

    // Values v1 = {1.1, 3.2, 4.1};

    // Eigen::Map<const Eigen::VectorXd> v(u.data(), u.size());
    // vec udual(u);
    // vec qdual(q);
    // dual pos = x;

    ValuesWrapper uw(u);
    ValuesWrapper qw(q);

    double sigma = sigmaVec[i](uw, qw, x, t).val;
    return sigma;
}
Value AutodiffWrapper::Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t) { return 0; }

// We need derivatives of the flux functions
Value AutodiffWrapper::dSigmaFn_du(Index i, Index j, const Values &u, const Values &q, Position x, Time t)
{
    // dual arr[] = {1, 2, 3, 4};
    // Eigen::Map<VectorXdual> v(arr, 4);

    ValuesWrapper uw(u);
    ValuesWrapper qw(q);
    std::cout << uw(j) << std::endl;
    auto dsdu = gradient(sigmaVec[i], wrt(uw(j)), at(uw, qw, x, t));
    return (Value)dsdu(0);
}
Value AutodiffWrapper::dSigmaFn_dq(Index i, Index j, const Values &u, const Values &q, Position x, Time t)
{
    // vec udual(u);
    // vec qdual(q);

    // auto dsdq = gradient(sigmaVec[i], wrt(qdual(j)), at(udual, qdual, x, t));
    // return (Value)dsdq(0);
    return 0;
}

// and for the sources
Value AutodiffWrapper::dSources_du(Index i, Index j, const Values &u, const Values &q, Position x, Time t) { return 0; }
Value AutodiffWrapper::dSources_dq(Index i, Index j, const Values &u, const Values &q, Position x, Time t) { return 0; }
Value AutodiffWrapper::dSources_dsigma(Index i, Index j, const Values &u, const Values &q, Position x, Time t) { return 0; }

// and initial conditions for u & q
Value AutodiffWrapper::InitialValue(Index i, Position x) const { return 0; }
Value AutodiffWrapper::InitialDerivative(Index i, Position x) const { return 0; }
