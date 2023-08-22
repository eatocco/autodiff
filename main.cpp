// C++ includes
#include <iostream>
#include <cmath>
// autodiff include
// #include <autodiff/forward/real.hpp>
// #include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
// #include "sigma.hpp"
// #include "source.hpp"
#include "AutodiffWrapper.hpp"
#include "Types.hpp"
using namespace autodiff;

int main()
{

    Position x = 5.0;

    Time t = 1.0;

    Index i = 0;
    Index j = 1;

    Values u = {2.0, 2.0, 3.1};
    Values q = {3.1, 5.4, 1.5};

    AutodiffWrapper a;

    double sigma = a.SigmaFn(i, u, q, x, t);
    double dsigmadu = a.dSigmaFn_du(i, j, u, q, x, t);
    std::cout << "sigma = " << sigma << std::endl;
    std::cout << "dsigmadu = " << dsigmadu << std::endl;

    // Position x = 5.0;
    // Time t = 1.0;
    // unsigned int sz = 3;

    // Value u(sz);
    // u << 1.1, 2.2, 1.5;
    // Value q(sz);
    // q << 4.1, 2.0, 1.3;

    // sigmaFn s1 = [](Value u, Value q, Position x, Time t)
    // {
    //     return (u * u).sum();
    // };
    // sigmaFn s2 = [](Value u, Value q, Position x, Time t)
    // {
    //     return 5 * q(1);
    // };
    // sigmaFn s3 = [](Value u, Value q, Position x, Time t)
    // {
    //     return (q * q).sum() + cos(x);
    // };

    // sigmaFnArray s = {s1, s2, s3};

    // FluxObject sigma(s);

    // std::cout
    //     << "sigma = " << sigma(u, q, x, t, 0) << std::endl;
    // std::cout << "dsigma/du = " << sigma.dq(u, q, x, t, 1, 1) << std::endl;

    // // testing using autodiff for a source term

    // sln solution = [](dual x, dual t)
    // {
    //     double uL = 1;
    //     double uR = 0.5;

    //     double xL = 0.1;
    //     double xR = 1;
    //     double k = 0.5;

    //     double a = (asinh(uL) - asinh(uR)) / (xL - xR);
    //     double b = (asinh(uL) - xL / xR * asinh(uR)) / (a * (xL / xR - 1));
    //     double c = (M_PI / 2 - 3 * M_PI / 2) / (xL - xR);
    //     double d = (M_PI / 2 - xL / xR * (3 * M_PI / 2)) / (c * (xL / xR - 1));

    //     dual ans = sinh(a * (x - b)) - cos(c * (x - d)) * exp(-k * t);
    //     return ans;
    // };

    // diffeq prob = [](sln &u, dual x, dual t)
    // {
    //     dual ux = derivative(u, wrt(x), at(x, t));
    //     dual ut = derivative(u, wrt(t), at(x, t));
    //     // burgers equation w/ source
    //     return ut + u(x, t) * ux;
    // };

    // SourceObj source(solution, prob);
    // std::cout << "u at x=0.5,t=5: " << source.u(0.5, 5) << std::endl;
    // std::cout << "Source at x=0.5,t=5: " << source(0.5, 5) << std::endl;

    return 0;
}
