#pragma once
#include <eigen3/Eigen/Core>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>

typedef autodiff::ArrayXreal Value;
typedef unsigned int Index;
typedef double Time;
typedef autodiff::real Position;

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct Params
{
    Position x;
    Time t;
    Value u;
    Value q;
    Index i;
    Params(Position x, Time t, Value u, Value q, Index i) : x(x), t(t), u(u), q(q), i(i){};
    Params() = default;

    //  Params(unsigned int sz, Position x, Time t, Index i, Index j) : x(x), t(t), i(i), j(j) { u(sz), q(sz); };
};

// typedef std::function<autodiff::dual(Position, Time &, Params &)> sigmaFn;
typedef std::function<autodiff::real(Value u, Value q, Position x, Time t)> sigmaFn;
typedef std::vector<sigmaFn> sigmaFnArray;

namespace autodiff
{

    class FluxObject
    {
    public:
        FluxObject() = default;
        //    FluxObject(sigmaFn fluxFn) : sigma(fluxFn) {}
        FluxObject(sigmaFnArray fluxFnArray) : sigmaVec(fluxFnArray) {}
        FluxObject(sigmaFnArray fluxFnArray, Value &u, Value &q, Position &x, Time &t) : sigmaVec(fluxFnArray), u(u), q(q), x(x), t(t) {}

        real operator()(Index i)
        {
            return sigmaVec[i](u, q, x, t);
        }
        real operator()(Value u, Value q, Position x, Time t, Index i)
        {
            return sigmaVec[i](u, q, x, t);
        }
        VectorXd du(Index i, Index j)
        {
            return gradient(sigmaVec[i], wrt(u(j)), at(u, q, x, t));
        }
        VectorXd du(Value u, Value q, Position x, Time t, Index i, Index j)
        {
            return gradient(sigmaVec[i], wrt(u(j)), at(u, q, x, t));
        }
        VectorXd dq(Index i, Index j)
        {
            return gradient(sigmaVec[i], wrt(q(j)), at(u, q, x, t));
        }
        VectorXd dq(Value u, Value q, Position x, Time t, Index i, Index j)
        {
            return gradient(sigmaVec[i], wrt(q(j)), at(u, q, x, t));
        }

    private:
        // sigmaFn sigma;
        sigmaFnArray sigmaVec;
        Value u;
        Value q;
        Position x;
        Time t;
    };
}