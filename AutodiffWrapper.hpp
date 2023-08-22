#ifndef AUTODIFFWRAPPER_HPP
#define AUTODIFFWRAPPER_HPP

#include "TransportSystem.hpp"
#include "toml11/toml.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using autodiff::dual;
using autodiff::VectorXdual;

struct vec
{
    VectorXdual values;
    operator VectorXdual() { return values; }
    auto &operator()(Index i)
    {
        return values(i);
    }
    vec(const Values &u) : values(u.size())
    {
        for (int i = 0; i < u.size(); i++)
            values(i) = u[i];
    }
};

typedef std::function<dual(VectorXdual u, VectorXdual q, dual x, double t)> sigmaFn;
typedef std::vector<sigmaFn> sigmaFnArray;

class AutodiffWrapper : public TransportSystem
{
public:
    AutodiffWrapper();
    // explicit AutodiffWrapper(toml::value const &config);
    //  Function for passing boundary conditions to the solver
    Value LowerBoundary(Index i, Time t) const override;
    Value UpperBoundary(Index i, Time t) const override;

    bool isLowerBoundaryDirichlet(Index i) const override;
    bool isUpperBoundaryDirichlet(Index i) const override;

    // The same for the flux and source functions -- the vectors have length nVars
    Value SigmaFn(Index i, const Values &u, const Values &q, Position x, Time t) override;
    Value Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t) override;

    // We need derivatives of the flux functions
    Value dSigmaFn_du(Index i, Index j, const Values &u, const Values &q, Position x, Time t) override;
    Value dSigmaFn_dq(Index i, Index j, const Values &u, const Values &q, Position x, Time t) override;

    // and for the sources
    Value dSources_du(Index i, Index j, const Values &u, const Values &q, Position x, Time t) override;
    Value dSources_dq(Index i, Index j, const Values &u, const Values &q, Position x, Time t) override;
    Value dSources_dsigma(Index i, Index j, const Values &u, const Values &q, Position x, Time t) override;

    // and initial conditions for u & q
    Value InitialValue(Index i, Position x) const override;
    Value InitialDerivative(Index i, Position x) const override;

private:
    sigmaFnArray sigmaVec;
};
#endif