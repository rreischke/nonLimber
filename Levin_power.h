#ifndef LEVIN_POWER_H
#define LEVIN_POWER_H

#include <vector>
#include <numeric>
#include <omp.h>
#include <string>
#include <sstream>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>

class Levin_power
{
private:
    static const double min_interval;
    static const double tol_abs;
    static const double tol_rel;
    static const double limber_tolerance;
    static const double min_sv;
    static const uint min_ell_check_Limber = 50;
    static const uint N_interp = 200;
    static const uint N_thread_max = 64; //modify this if needed (number of threads)
    gsl_spline2d *spline_P_l;
    gsl_spline2d *spline_P_nl;
    gsl_interp_accel *acc_P_l_k;
    gsl_interp_accel *acc_P_l_z;
    gsl_interp_accel *acc_P_nl_k;
    gsl_interp_accel *acc_P_nl_z;
    std::vector<gsl_interp_accel *> acc_Weight;
    std::vector<gsl_spline *> spline_Weight;

    gsl_interp_accel *acc_chi_of_z;
    gsl_spline *spline_chi_of_z;
    gsl_interp_accel *acc_z_of_chi;
    gsl_spline *spline_z_of_chi;

    std::vector<double> kernel_maximum;

    std::vector<gsl_interp_accel *> acc_aux_kernel;
    std::vector<gsl_spline *> spline_aux_kernel;
    std::vector<double> aux_kmax;
    std::vector<double> aux_kmin;

    uint d;
    uint n_total;
    uint number_counts;

    double chi_min, chi_max;
    double k_min, k_max;

    uint *integration_variable_i_tomo, *integration_variable_j_tomo;

    uint *integration_variable_Limber_ell, *integration_variable_Limber_i_tomo, *integration_variable_Limber_j_tomo;
    bool *integration_variable_Limber_linear;

    double gslIntegratecquad(double (*fc)(double, void *), double a, double b);

public:
    Levin_power(uint number_count, std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk_l, std::vector<double> pk_nl);

    ~Levin_power();

    uint find_kernel_maximum(std::vector<double> kernel);

    void init_splines(std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk_l, std::vector<double> pk_nl);

    double kernel(double chi, uint i_tomo);

    double chi_of_z(double z);

    double z_of_chi(double chi);

    double power_linear(double z, double k);

    double power_nonlinear(double z, double k);

    double w(double chi, double k, uint ell, uint i);

    double A_matrix(uint i, uint j, double chi, double k, uint ell);

    void allocate();

    void free();

    std::vector<double> setNodes(double A, double B, uint col);

    double basis_function(double A, double B, double x, uint m);

    double basis_function_prime(double A, double B, double x, uint m);

    double F_linear(double chi, uint i_tomo, double k);

    double F_nonlinear(double chi, uint i_tomo, double k);

    std::vector<double> solve_LSE(double A, double B, uint col, std::vector<double> x_j, uint i_tomo, double k, uint ell, bool linear);

    double p(double A, double B, uint i, double x, uint col, std::vector<double> c);

    double integrate(double A, double B, uint col, uint i_tomo, double k, uint ell, bool linear);

    double iterate(double A, double B, uint col, uint i_tomo, double k, uint ell, uint smax, bool verbose, bool linear);

    uint findMax(const std::vector<double> vec);

    std::vector<double> linear_spaced(double min, double max, uint N);

    double levin_integrate_bessel(double k, uint ell, uint i_tomo, bool linear);

    double Limber(uint ell, uint i_tomo, uint j_tomo, bool linear);

    static double Limber_kernel(double, void *);

    void set_auxillary_splines(std::vector<uint> use_limber, uint ell, bool linear);

    void free_auxillary_splines();

    double auxillary_weight(uint i_tomo, double k);

    static double k_integration_kernel(double, void *);

    double C_ell_full(uint i_tomo, uint j_tomo);

    std::vector<double> all_C_ell(std::vector<uint> ell, bool linear);
};

#endif