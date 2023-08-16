#ifndef SOURCE_HPP
#define SOURCE_HPP

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

namespace autodiff
{

    typedef std::function<dual(dual x, dual t)> sln;
    typedef std::function<dual(sln &u, dual x, dual t)> diffeq;

    class SourceObj
    {
    public:
        SourceObj(sln &solution, diffeq &prob) : solution(solution), problem(prob) {}

        dual operator()(dual x, double t)
        {
            return problem(solution, x, t);
        }

        dual u(dual x, dual t)
        {
            return solution(x, t);
        }

    private:
        sln solution;
        diffeq problem;
    };
}

#endif