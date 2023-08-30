#include "Autodiff3VarCyl.hpp"
#include "Constants.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
#include <cmath>
using namespace autodiff;

// Autodiff3VarCyl::Autodiff3VarCyl(toml::value const &config)
// {
//     nVars = 3;

//     uR(nVars);
//     uL(nVars);

//     if (config.count("Autodiff3VarCyl") != 1)
//         throw std::invalid_argument("There should be a [Autodiff3VarCyl] section if you are using the Autodiff3VarCyl physics model.");

//     auto const &InternalConfig = config.at("Autodiff3VarCyl");

//     x_L = toml::find_or(InternalConfig, "x_L", 0.1);
//     x_R = toml::find_or(InternalConfig, "x_R", 1.0);

//     isUpperDirichlet = toml::find_or(InternalConfig, "isUpperDirichlet", true);
//     isLowerDirichlet = toml::find_or(InternalConfig, "isLowerDirichlet", true);

//     double nL = toml::find_or(InternalConfig, "nL", 3e18);
//     double nR = toml::find_or(InternalConfig, "nR", 4e18);

//     double peL = toml::find_or(InternalConfig, "peL", 2); // keV
//     double peR = toml::find_or(InternalConfig, "peR", 2); // keV

//     double piL = toml::find_or(InternalConfig, "piL", 2); // keV
//     double piR = toml::find_or(InternalConfig, "piR", 2); // keV

//     uR << nR, peR, piR;
//     uL << nL, peL, piL;
// }

Value Autodiff3VarCyl::LowerBoundary(Index i, Time t) const { return uL(i); }

Value Autodiff3VarCyl::UpperBoundary(Index i, Time t) const { return uR(i); }

bool Autodiff3VarCyl::isLowerBoundaryDirichlet(Index i) const { return isLowerDirichlet; }

bool Autodiff3VarCyl::isUpperBoundaryDirichlet(Index i) const { return isUpperDirichlet; }

// The same for the flux and source functions -- the vectors have length nVars

Value Autodiff3VarCyl::SigmaFn(Index i, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    Value sigma = sigmaVec[i](uw, qw, x, t).val;
    return sigma;
}
Value Autodiff3VarCyl::Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t)
{

    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(sigma);
    Value S = SourceVec[i](uw, qw, sw, x, t).val;

    if (isTestProblem)
        S += TestSource(i, x, t);
    return S;
}

// We need derivatives of the flux functions
void Autodiff3VarCyl::dSigmaFn_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(sigmaVec[i], wrt(uw), at(uw, qw, x, t));
}
void Autodiff3VarCyl::dSigmaFn_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{

    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(sigmaVec[i], wrt(qw), at(uw, qw, x, t));
}

// and for the sources
void Autodiff3VarCyl::dSources_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(uw), at(uw, qw, sw, x, t));
}
void Autodiff3VarCyl::dSources_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(qw), at(uw, qw, sw, x, t));
}
void Autodiff3VarCyl::dSources_dsigma(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(sw), at(uw, qw, sw, x, t));
}

// and initial conditions for u & q
Value Autodiff3VarCyl::InitialValue(Index i, Position x) const
{
    if (isTestProblem)
    {
        double sol = TestSols[i](x, 0).val.val;
        return sol; // TestSols[i](x, 0)(0).val;
    }

    else
        return 0;
}
Value Autodiff3VarCyl::InitialDerivative(Index i, Position x) const
{

    if (isTestProblem)
    {
        dual2nd pos = x;
        dual2nd t = 0.0;
        double deriv = derivative(TestSols[i], wrt(pos), at(pos, t));
        return deriv;
    }
    else
    {
        return 0;
    }
}
double Autodiff3VarCyl::TestSource(Index i, Position x, Time t)
{

    dual2nd T = t;
    dual2nd pos = x;
    double ut = derivative(TestSols[i], wrt(T), at(pos, T));
    nVars = 3;
    VectorXdual q(nVars);
    VectorXdual u(nVars);

    VectorXdual dq(nVars);
    VectorXdual sigma(nVars);

    for (Index j = 0; j < 3; j++)
    {
        auto [q0, q1, q2] = derivatives(TestSols[j], wrt(pos, pos), at(pos, T));

        u(j) = q0;
        q(j) = q1;
        dq(j) = q2;
    }

    for (Index j = 0; j < nVars; j++)
    {
        sigma(j) = sigmaVec[j](u, q, x, t);
    }

    Values ugrad = gradient(sigmaVec[i], wrt(u), at(u, q, x, t));
    Values qgrad = gradient(sigmaVec[i], wrt(q), at(u, q, x, t));
    dual xdual = x;
    Values xgrad = gradient(sigmaVec[i], wrt(xdual), at(u, q, xdual, t));
    double uxd = xgrad(0);

    for (Index j = 0; j < nVars; j++)
    {
        uxd += ugrad(j) * q(j).val + qgrad(j) * dq(j).val;
    }
    double S = SourceVec[i](u, q, sigma, x, t).val;
    double St = ut + uxd - S;

    return St;
}

const double B_mid = 0.3; // Tesla
const double Om_i = e_charge * B_mid / ionMass;
const double Om_e = e_charge * B_mid / electronMass;
const double lambda = 15.0;

dual tau_i(dual n, dual Pi)
{
    if (Pi > 0)
        return ::sqrt(2) * 3.44e11 * (1.0 / pow(n, 5.0 / 2.0)) * (pow(Pi / e_charge, 3.0 / 2.0)) * (1.0 / lambda) * (::sqrt(ionMass / electronMass));
    else
        return ::sqrt(2) * 3.44e11 * (1.0 / n) * (pow(n, 5.0 / 2.0)) * (1.0 / lambda) * (::sqrt(ionMass / electronMass)); // if we have a negative temp just treat it as 1eV
}

dual tau_e(dual n, dual Pe)
{
    if (Pe > 0)
        return 3.44e11 * (1.0 / pow(n, 5.0 / 2.0)) * (pow(Pe / e_charge, 3.0 / 2.0)) * (1.0 / lambda);
    else
        return 3.44e11 * (1.0 / pow(n, 5.0 / 2.0)) * (pow(n, 3.0 / 2.0)) * (1.0 / lambda);
}

sigmaFn Gamma = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    // maybe add a factor of sqrt x if x = r^2/2

    dual G = 2 * x * u(1) / (electronMass * Om_e * Om_e * tau_e(u(0), u(1))) * ((q(1) / 2 - q(2)) / u(1) + 3. / 2. * q(0) / u(0));
    return G;
};

sigmaFn qi = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma(u, q, x, t);
    dual kappa = 2. * u(2) / (ionMass * Om_i * Om_i * tau_i(u(0), u(2)));
    dual qri = -kappa * u(2) / u(0) * (q(2) / u(2) - q(0) / u(0));
    dual Q = (2. / 3.) * ((5. / 2.) * u(2) / u(0) * G + 2. * x * qri);
    return Q;
};
sigmaFn qe = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma(u, q, x, t);
    dual kappa = 4.66 * u(1) / (electronMass * Om_e * Om_e * tau_e(u(0), u(1)));
    dual qre = -kappa * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) + (3. / 2.) * u(1) / (u(0) * electronMass * Om_e * Om_e * tau_e(u(0), u(1))) * (q(2) + q(1));
    dual Q = (2. / 3.) * (5. / 2. * u(1) / u(0) * G + (2. * x) * qre);
    return Q;
};
sigmaFnArray Autodiff3VarCyl::sigmaVec = {Gamma, qe, qi};

SourceFn Sn = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return 0;
};

// look at ion and electron sources again -- they should be opposite
SourceFn Spi = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual G = Gamma(u, q, x, t) / (2. * x);
    dual V = G / u(0);
    dual S = 2. / 3. * sqrt(2. * x) * V * q(2);
    return S;
};
SourceFn Spe = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual G = Gamma(u, q, x, t) / (2. * x);
    dual V = G / u(0);
    dual S = 2. / 3. * sqrt(2. * x) * V * q(1);
    return S;
};

SourceFnArray Autodiff3VarCyl::SourceVec = {Sn, Spe, Spi};

Solution Autodiff3VarCyl::Tn = [](dual2nd x, dual2nd t)
{
    double uL = 3e18;

    double uR = 4e18;

    double xL = 0.1;
    double xR = 1;
    double k = 0.5;

    dual2nd a = (asinh(uL) - asinh(uR)) / (xL - xR);
    dual2nd b = (asinh(uL) - xL / xR * asinh(uR)) / (a * (xL / xR - 1));
    dual2nd c = (M_PI / 2 - 3 * M_PI / 2) / (xL - xR);
    dual2nd d = (M_PI / 2 - xL / xR * (3 * M_PI / 2)) / (c * (xL / xR - 1));

    dual2nd u = sinh(a * (x - b)) - cos(c * (x - d)) * uL * exp(-k * t) * exp(-0.5 * x * x);

    return u;
};
Solution Autodiff3VarCyl::Tpi = [](dual2nd x, dual2nd t)
{
    double uL = e_charge * 3e18 * 1000;

    double uR = e_charge * 4e18 * 1000;

    double xL = 0.1;
    double xR = 1;
    double k = 0.5;

    dual2nd a = (asinh(uL) - asinh(uR)) / (xL - xR);
    dual2nd b = (asinh(uL) - xL / xR * asinh(uR)) / (a * (xL / xR - 1));
    dual2nd c = (M_PI / 2 - 3 * M_PI / 2) / (xL - xR);
    dual2nd d = (M_PI / 2 - xL / xR * (3 * M_PI / 2)) / (c * (xL / xR - 1));

    dual2nd u = sinh(a * (x - b)) - cos(c * (x - d)) * uL * exp(-k * t) * exp(-0.5 * x * x);

    return u;
};
Solution Autodiff3VarCyl::Tpe = [](dual2nd x, dual2nd t)
{
    double uL = e_charge * 3e18 * 1000;

    double uR = e_charge * 4e18 * 1000;

    double xL = 0.1;
    double xR = 1;
    double k = 0.5;

    dual2nd a = (asinh(uL) - asinh(uR)) / (xL - xR);
    dual2nd b = (asinh(uL) - xL / xR * asinh(uR)) / (a * (xL / xR - 1));
    dual2nd c = (M_PI / 2 - 3 * M_PI / 2) / (xL - xR);
    dual2nd d = (M_PI / 2 - xL / xR * (3 * M_PI / 2)) / (c * (xL / xR - 1));

    dual2nd u = sinh(a * (x - b)) - cos(c * (x - d)) * uL * exp(-k * t) * exp(-0.5 * x * x);

    return u;
};

TestSolVec Autodiff3VarCyl::TestSols = {Tn, Tpe, Tpi};
