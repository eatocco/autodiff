// C++ includes
#include <iostream>
#include <cmath>
// autodiff include
// #include <autodiff/forward/real.hpp>
// #include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
// #include "sigma.hpp"
// #include "source.hpp"
#include "Autodiff3VarCyl.hpp"
#include "Types.hpp"
using namespace autodiff;

int main()
{

    Position x = 0.4;

    Time t = 0.0;

    Index i = 0;
    Index j = 0;

    Values u(3);
    u << 5700253868753132215.8273104052019, 913.28136977957033286570608431802, 913.2813697795703328657060843;
    Values q(3);
    q << 4929545543581451975.3046964112213, 789.80029571649370225130951430966, 789.80029571649370225130951430966;
    Values grad(3);
    // = {0.0, 0.0, 0.0};
    Autodiff3VarCyl a;

    // double sigma = a.SigmaFn(i, u, q, x, t);
    // a.dSigmaFn_dq(i, grad, u, q, x, t);
    std::cout << "Gamma = " << a.SigmaFn(0, u, q, x, t) << std::endl;
    std::cout << "Peflux = " << a.SigmaFn(1, u, q, x, t) << std::endl;
    std::cout << "Piflux = " << a.SigmaFn(2, u, q, x, t) << std::endl;
    // std::cout << "dsigmadq = " << grad << std::endl;
    double St = a.TestSource(0, x, 0.0);
    std::cout << "Test source for density = " << St << std::endl;
    double Stpe = a.TestSource(1, x, 0.0);
    std::cout << "Test source for pe = " << Stpe << std::endl;
    double Stpi = a.TestSource(2, x, 0.0);
    std::cout << "Test source for pi = " << Stpi << std::endl;

    // dual2nd d = 3.0;
    // auto n = d.val.val;
    // std::cout << n << std::endl;
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
