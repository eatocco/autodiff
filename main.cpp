// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include "sigma.hpp"
using namespace autodiff;

int main()
{
    Position x = 5.0;
    Time t = 1.0;
    int sz = 3;
    //  Params p(sz, x, t, 1, 2);

    Value u(sz);
    u << 1.1, 2.2, 1.5;
    Value q(sz);
    q << 4.1, 2.0, 1.3;

    sigmaFn s1 = [](Value u, Value q, Position x, Time t)
    {
        return (u * u).sum();
    };
    sigmaFn s2 = [](Value u, Value q, Position x, Time t)
    {
        return 5 * q(1);
    };
    sigmaFn s3 = [](Value u, Value q, Position x, Time t)
    {
        return (q * q).sum() + cos(x);
    };

    sigmaFnArray s = {s1, s2, s3};

    FluxObject sigma(s, u, q, x, t);

    std::cout
        << "sigma = " << sigma(0) << std::endl;
    std::cout << "du/dx = " << sigma.du(0, 0) << std::endl;
    return 0;
}
