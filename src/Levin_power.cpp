#include "Levin_power.h"
#include <fstream>

const double Levin_power::min_interval = 1.e-2;
const double Levin_power::limber_tolerance = 1.0e-2;
const double Levin_power::tol_abs = 1.0e-30;
const double Levin_power::tol_rel = 1.0e-7;
const double Levin_power::min_sv = 1.0e-10;
const double Levin_power::kernel_overlap_eps = 5e-6;

Levin_power::Levin_power(uint number_count, std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk_l, std::vector<double> pk_nl)
{
    if (kernel.size() != chi_cl.size())
    {
        throw std::range_error("kernel dimension does not match size of chi_cl");
    }
    if (pk_nl.size() != z_pk.size() * k_pk.size())
    {
        throw std::range_error("Pk_nl dimension does not match sizes of z_pk and k_pk");
    }
    d = 2;
    n_total = kernel.at(0).size();
    number_counts = number_count;
    init_splines(z_bg, chi_bg, chi_cl, kernel, k_pk, z_pk, pk_l, pk_nl);
    check_kernel_overlap();
    tables_result_set = false;
}

Levin_power::~Levin_power()
{
    delete integration_variable_i_tomo;
    delete integration_variable_j_tomo;
    delete integration_variable_Limber_ell;
    delete integration_variable_Limber_i_tomo;
    delete integration_variable_Limber_j_tomo;
    delete integration_variable_extended_Limber_ell;
    delete integration_variable_extended_Limber_i_tomo;
    delete integration_variable_extended_Limber_j_tomo;
    delete integration_variable_Limber_linear;
    delete integration_variable_norm_kernel_i_tomo;
    delete integration_variable_norm_kernel_overlap_i_tomo;
    delete integration_variable_norm_kernel_overlap_j_tomo;
    gsl_spline_free(spline_z_of_chi);
    gsl_interp_accel_free(acc_z_of_chi);
    gsl_spline_free(spline_chi_of_z);
    gsl_interp_accel_free(acc_chi_of_z);
    for (uint i = 0; i < n_total; i++)
    {
        gsl_spline_free(spline_Weight.at(i));
        gsl_interp_accel_free(acc_Weight.at(i));
    }
    gsl_spline2d_free(spline_P_l);
    gsl_interp_accel_free(acc_P_l_k);
    gsl_interp_accel_free(acc_P_l_z);
    gsl_spline2d_free(spline_d2P_d2k);
    gsl_interp_accel_free(acc_d2P_d2k_k);
    gsl_interp_accel_free(acc_d2P_d2k_z);
    gsl_spline2d_free(spline_P_nl);
    gsl_interp_accel_free(acc_P_nl_k);
    gsl_interp_accel_free(acc_P_nl_z);
    for (uint i = 0; i < n_total; i++)
    {
        gsl_spline_free(spline_aux_kernel.at(i));
        gsl_interp_accel_free(acc_aux_kernel.at(i));
    }
}

uint Levin_power::find_kernel_maximum(std::vector<double> kernel)
{
    uint result = 0.0;
    for (uint i = 0; i < kernel.size(); i++)
    {
        if (kernel.at(i + 1) < kernel.at(i))
        {
            result = i;
            break;
        }
    }
    return result;
}

void Levin_power::init_splines(std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk_l, std::vector<double> pk_nl)
{
    integration_variable_i_tomo = new uint[N_thread_max];
    integration_variable_j_tomo = new uint[N_thread_max];
    integration_variable_Limber_i_tomo = new uint[N_thread_max];
    integration_variable_Limber_j_tomo = new uint[N_thread_max];
    integration_variable_Limber_ell = new uint[N_thread_max];
    integration_variable_extended_Limber_i_tomo = new uint[N_thread_max];
    integration_variable_extended_Limber_j_tomo = new uint[N_thread_max];
    integration_variable_extended_Limber_ell = new uint[N_thread_max];
    integration_variable_Limber_linear = new bool[N_thread_max];
    integration_variable_norm_kernel_i_tomo = new uint[N_thread_max];
    integration_variable_norm_kernel_overlap_i_tomo = new uint[N_thread_max];
    integration_variable_norm_kernel_overlap_j_tomo = new uint[N_thread_max];
    spline_chi_of_z = gsl_spline_alloc(gsl_interp_steffen, z_bg.size());
    spline_z_of_chi = gsl_spline_alloc(gsl_interp_steffen, z_bg.size());
    acc_chi_of_z = gsl_interp_accel_alloc();
    acc_z_of_chi = gsl_interp_accel_alloc();
    gsl_spline_init(spline_chi_of_z, &z_bg[0], &chi_bg[0], z_bg.size());
    gsl_spline_init(spline_z_of_chi, &chi_bg[0], &z_bg[0], z_bg.size());
    for (uint i_tomo = 0; i_tomo < kernel.at(0).size(); i_tomo++)
    {
        spline_Weight.push_back(gsl_spline_alloc(gsl_interp_steffen, chi_cl.size()));
        acc_Weight.push_back(gsl_interp_accel_alloc());
        std::vector<double> init_weight(chi_cl.size(), 0.0);
        for (uint i = 0; i < chi_cl.size(); i++)
        {
            init_weight.at(i) = kernel.at(i).at(i_tomo);
        }
        kernel_maximum.push_back(chi_cl.at(find_kernel_maximum(init_weight)));
        gsl_spline_init(spline_Weight.at(i_tomo), &chi_cl[0], &init_weight[0], chi_cl.size());
    }
    chi_min = chi_cl.at(1);
    chi_max = chi_cl.at(chi_cl.size() - 2);
    for (uint i_tomo = 0; i_tomo < kernel.at(0).size(); i_tomo++)
    {
        uint test_of_zero = 0;
        double chi1 = 10000;
        double chi2;
        for (uint i = 0; i < chi_cl.size(); i++)
        {
            if (kernel.at(i).at(i_tomo) > 1e-15 && test_of_zero == 0)
            {
                chi1 = chi_cl.at(i);
                test_of_zero = 1;
            }
            if (chi_cl.at(i) > chi1 && kernel.at(i).at(i_tomo) < 1e-15 && test_of_zero == 1)
            {
                chi2 = chi_cl.at(i);
                break;
            }
        }
        bessel_high.push_back(chi2);
        bessel_low.push_back(chi1);
    }

    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    acc_P_l_k = gsl_interp_accel_alloc();
    acc_P_l_z = gsl_interp_accel_alloc();
    acc_P_nl_k = gsl_interp_accel_alloc();
    acc_P_nl_z = gsl_interp_accel_alloc();
    acc_d2P_d2k_k = gsl_interp_accel_alloc();
    acc_d2P_d2k_z = gsl_interp_accel_alloc();
    spline_P_l = gsl_spline2d_alloc(T, k_pk.size(), z_pk.size());
    spline_P_nl = gsl_spline2d_alloc(T, k_pk.size(), z_pk.size());
    spline_d2P_d2k = gsl_spline2d_alloc(T, k_pk.size(), z_pk.size());
    k_min = k_pk.at(1);
    k_max = k_pk.at(k_pk.size() - 2);
    for (uint i = 0; i < k_pk.size(); i++)
    {
        k_pk.at(i) = log(k_pk.at(i));
        for (uint j = 0; j < z_pk.size(); j++)
        {
            pk_l.at(i * z_pk.size() + j) = log(pk_l.at(i * z_pk.size() + j));
            pk_nl.at(i * z_pk.size() + j) = log(pk_nl.at(i * z_pk.size() + j));
        }
    }
    gsl_spline2d_init(spline_P_l, &k_pk[0], &z_pk[0], &pk_l[0], k_pk.size(), z_pk.size());
    gsl_spline2d_init(spline_P_nl, &k_pk[0], &z_pk[0], &pk_nl[0], k_pk.size(), z_pk.size());
    std::vector<double> init_d2P_d2k(k_pk.size() * z_pk.size());
    for (uint i = 0; i < k_pk.size(); i++)
    {
        for (uint j = 0; j < z_pk.size(); j++)
        {
            init_d2P_d2k.at(j * z_pk.size() + i) = d2P_d2k(exp(k_pk.at(i)), z_pk.at(j));
        }
    }
    gsl_spline2d_init(spline_d2P_d2k, &k_pk[0], &z_pk[0], &init_d2P_d2k[0], k_pk.size(), z_pk.size());
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        aux_kmax.push_back(0.0);
        aux_kmin.push_back(0.0);
    }
    for (uint i = 0; i < n_total; i++)
    {
        spline_aux_kernel.push_back(gsl_spline_alloc(gsl_interp_steffen, N_interp));
        acc_aux_kernel.push_back(gsl_interp_accel_alloc());
    }
    factor.push_back(1.0);
    for (uint i = 1; i < 5000; i++)
    {
        factor.push_back(sqrt((i + 2.0) * (i + 1.0) * i * (i - 1.0)));
    }
}

void Levin_power::check_kernel_overlap()
{
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        kernel_norm.push_back(1.0);
    }

    for (uint i_tomo = 0; i_tomo < n_total * n_total; i_tomo++)
    {
        kernel_overlap.push_back(true);
    }
#pragma omp parallel for
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        uint tid = omp_get_thread_num();
        integration_variable_norm_kernel_i_tomo[tid] = i_tomo;
        kernel_norm.at(i_tomo) = gslIntegrateqag(normalize_kernels_kernel, chi_min, chi_max);
    }
#pragma omp parallel for
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        uint tid = omp_get_thread_num();
        for (uint j_tomo = i_tomo; j_tomo < n_total; j_tomo++)
        {
            integration_variable_norm_kernel_overlap_i_tomo[tid] = i_tomo;
            integration_variable_norm_kernel_overlap_j_tomo[tid] = j_tomo;
            double overlap = gslIntegrateqag(kernels_overlap_kernel, chi_min, chi_max);
            if (overlap < kernel_overlap_eps)
            {
                kernel_overlap.at(i_tomo * n_total + j_tomo) = false;
            }
        }
    }
    for (uint i = 0; i < n_total; i++)
    {
        ell_eLimber.push_back(ellmax_non_limber);
    }
}

double Levin_power::kernels(double chi, uint i_tomo)
{
    return gsl_spline_eval(spline_Weight.at(i_tomo), chi, acc_Weight.at(i_tomo));
}

double Levin_power::kernel_normed(double chi, uint i_tomo)
{
    return kernels(chi, i_tomo) / kernel_norm.at(i_tomo);
}

double Levin_power::normalize_kernels_kernel(double chi, void *p)
{
    uint tid = omp_get_thread_num();
    Levin_power *lp = static_cast<Levin_power *>(p);
    return lp->kernel_normed(chi, lp->integration_variable_norm_kernel_i_tomo[tid]);
}

double Levin_power::kernels_overlap_kernel(double chi, void *p)
{
    uint tid = omp_get_thread_num();
    Levin_power *lp = static_cast<Levin_power *>(p);
    return lp->kernel_normed(chi, lp->integration_variable_norm_kernel_overlap_i_tomo[tid]) * lp->kernel_normed(chi, lp->integration_variable_norm_kernel_overlap_j_tomo[tid]);
}

double Levin_power::chi_of_z(double z)
{
    return gsl_spline_eval(spline_chi_of_z, z, acc_chi_of_z);
}

double Levin_power::z_of_chi(double chi)
{
    return gsl_spline_eval(spline_z_of_chi, chi, acc_z_of_chi);
}

double Levin_power::power_linear(double z, double k)
{
    return exp(gsl_spline2d_eval(spline_P_l, log(k), z, acc_P_l_k, acc_P_l_z));
}

double Levin_power::power_nonlinear(double z, double k)
{
    if (k >= k_min)
    {
        return exp(gsl_spline2d_eval(spline_P_nl, log(k), z, acc_P_nl_k, acc_P_nl_z));
    }
    else
    {
        return exp(gsl_spline2d_eval_deriv_x(spline_P_nl, log(k_min), z, acc_P_nl_k, acc_P_nl_z) * (log(k) - log(k_min)) + gsl_spline2d_eval(spline_P_nl, log(k_min), z, acc_P_nl_k, acc_P_nl_z));
    }
}

double Levin_power::w(double chi, double k, uint ell, uint i, bool strict)
{
    if (!strict)
    {
        switch (i)
        {
        case 0:
            return gsl_sf_bessel_jl(ell, chi * k);
        case 1:
            return gsl_sf_bessel_jl(ell - 1, chi * k);
        default:
            return 0.0;
        }
    }
    else
    {
        gsl_sf_result r;
        int status;
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_jl_e(ell, chi * k, &r);
            break;
        case 1:
            status = gsl_sf_bessel_jl_e(ell - 1, chi * k, &r);
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute spherical Bessel function for ell=" << ell - i << std::endl;
        }
        return r.val;
    }
}

double Levin_power::A_matrix(uint i, uint j, double chi, double k, uint ell)
{
    if (i == 0 && j == 0)
    {
        return -(ell + 1.0) / chi;
    }
    if (i * j == 1)
    {
        return (ell - 1.0) / chi;
    }
    if (i < j)
    {
        return k;
    }
    else
    {
        return -k;
    }
}

std::vector<double> Levin_power::setNodes(double A, double B, uint col)
{
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    for (uint j = 0; j < n; j++)
    {
        x_j[j] = A + j * (B - A) / (n - 1);
    }
    return x_j;
}

double Levin_power::basis_function(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 1.0;
    }
    return pow((x - (A + B) / 2) / (B - A), m);
}

double Levin_power::basis_function_prime(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 0.0;
    }
    if (m == 1)
    {
        return 1.0 / (B - A);
    }
    return m / (B - A) * pow((x - (A + B) / 2.) / (B - A), (m - 1));
}

double Levin_power::F_linear(double chi, uint i_tomo, double k)
{
    double z = z_of_chi(chi);
    if (i_tomo < number_counts)
    {
        return sqrt(power_linear(z, k)) * kernels(chi, i_tomo) * k;
    }
    else
    {
        return sqrt(power_linear(z, k)) * kernels(chi, i_tomo) / (gsl_pow_2(chi) * k);
    }
}

double Levin_power::F_nonlinear(double chi, uint i_tomo, double k)
{
    double z = z_of_chi(chi);
    if (i_tomo < number_counts)
    {
        return sqrt(power_nonlinear(z, k)) * kernels(chi, i_tomo) * k;
    }
    else
    {
        return sqrt(power_nonlinear(z, k)) * kernels(chi, i_tomo) / (gsl_pow_2(chi) * k);
    }
}

std::vector<double> Levin_power::solve_LSE(double A, double B, uint col, std::vector<double> x_j, uint i_tomo, double k, uint ell, bool linear)
{
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            if (linear)
            {
                gsl_vector_set(F_stacked, j, F_linear(x_j[j], i_tomo, k));
            }
            else
            {
                gsl_vector_set(F_stacked, j, F_nonlinear(x_j[j], i_tomo, k));
            }
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix(q, i, x_j[j], k, ell) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s;
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    int lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_permutation_free(P);
    gsl_set_error_handler(old_handler);
    gsl_matrix_free(matrix_G);
    return result;
}

double Levin_power::p(double A, double B, uint i, double x, uint col, std::vector<double> c)
{
    uint n = (col + 1) / 2;
    n *= 2;
    double result = 0.0;
    for (uint m = 0; m < n; m++)
    {
        result += c[i * n + m] * basis_function(A, B, x, m);
    }
    return result;
}

double Levin_power::integrate(double A, double B, uint col, uint i_tomo, double k, uint ell, bool linear)
{
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n);
    x_j = setNodes(A, B, col);
    c = solve_LSE(A, B, col, x_j, i_tomo, k, ell, linear);
    for (uint i = 0; i < d; i++)
    {
        result += p(A, B, i, B, col, c) * w(B, k, ell, i) - p(A, B, i, A, col, c) * w(A, k, ell, i);
    }
    return result;
}

double Levin_power::iterate(double A, double B, uint col, uint i_tomo, double k, uint ell, uint smax, bool verbose, bool linear)
{
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate(A, B, col / 2, i_tomo, k, ell, linear);
    double I_full = integrate(A, B, col, i_tomo, k, ell, linear);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        /*if (verbose) // print borders, approximations and error estimates for all subintervals
        {
            std::cerr << "estimate: " << std::scientific << result << std::endl
                      << sub << " subintervals: " << std::endl;
            for (uint i = 0; i < approximations.size(); ++i)
            {
                std::cerr << "[" << x_sub[i] << "," << x_sub[i + 1] << "]: " << approximations[i] << " (" << error_estimates[i] << ")" << std::endl;
                std::cerr << std::endl;
            }
        }*/
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs))
        {
            if (verbose)
            {
                std::cerr << "converged!" << std::endl;
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = findMax(error_estimates) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection!" << std::endl;
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        I_half = integrate(x_sub.at(i - 1), x_sub.at(i), col / 2, i_tomo, k, ell, linear);
        I_full = integrate(x_sub.at(i - 1), x_sub.at(i), col, i_tomo, k, ell, linear);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate(x_sub.at(i), x_sub.at(i + 1), col / 2, i_tomo, k, ell, linear);
        I_full = integrate(x_sub.at(i), x_sub.at(i + 1), col, i_tomo, k, ell, linear);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of subintervals reached!" << std::endl;
    }
    return result;
}

uint Levin_power::findMax(const std::vector<double> vec)
{
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

std::vector<double> Levin_power::linear_spaced(double min, double max, uint N)
{
    std::vector<double> result(N);
    for (uint i = 0; i < N; i++)
    {
        result.at(i) = min + (max - min) / (N - 1.0) * i;
    }
    return result;
}

double Levin_power::levin_integrate_bessel(double k, uint ell, uint i_tomo, bool linear)
{
    uint n_col = 7;
    uint n_sub = 10;
    gsl_set_error_handler_off();
    return iterate(chi_min, chi_max, n_col, i_tomo, k, ell, n_sub, false, linear);
}

void Levin_power::set_auxillary_splines(uint ell, bool linear)
{
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        std::vector<double> k_interp_i_tomo(N_interp);
        std::vector<double> I_bessel_i_tomo(N_interp);
        if (ell < ell_eLimber.at(i_tomo))
        {
            double kmin_i_tomo = GSL_MAX(0.5 * (ell + 0.5) / kernel_maximum.at(i_tomo), k_min);
            double kmax_i_tomo = GSL_MIN(8.0 * (ell + 0.5) / kernel_maximum.at(i_tomo), k_max);
            if (ell <= 5)
            {
                kmax_i_tomo *= 3.0;
            }
            if (ell > 40)
            {
                kmin_i_tomo = GSL_MAX(0.5 * (ell + 0.5) / kernel_maximum.at(i_tomo), k_min);
                kmax_i_tomo = GSL_MIN(8.0 * (ell + 0.5) / kernel_maximum.at(i_tomo), k_max);
            }
            if (i_tomo >= number_counts)
            {
                kmin_i_tomo = GSL_MAX(0.02 * (ell + 0.5) / kernel_maximum.at(i_tomo), k_min);
                kmax_i_tomo = GSL_MIN(100.0 * (ell + 0.5) / kernel_maximum.at(i_tomo), k_max);
            }
            k_interp_i_tomo = linear_spaced(log(kmin_i_tomo), log(kmax_i_tomo), N_interp);
#pragma omp parallel for
            for (uint i = 0; i < N_interp; i++)
            {
                I_bessel_i_tomo.at(i) = levin_integrate_bessel(exp(k_interp_i_tomo.at(i)), ell, i_tomo, linear);
            }
        }
        gsl_spline_init(spline_aux_kernel.at(i_tomo), &k_interp_i_tomo[0], &I_bessel_i_tomo[0], N_interp);
        aux_kmax.at(i_tomo) = exp(k_interp_i_tomo.at(k_interp_i_tomo.size() - 1));
        aux_kmin.at(i_tomo) = exp(k_interp_i_tomo.at(0));
    }
}

double Levin_power::Limber(uint ell, uint i_tomo, uint j_tomo, bool linear)
{
    double fac = 1.0;
    if (i_tomo >= number_counts)
    {
        fac /= gsl_pow_2(ell + 0.5);
    }
    if (j_tomo >= number_counts)
    {
        fac /= gsl_pow_2(ell + 0.5);
    }
    if (ell < 900)
    {
        return fac * extended_Limber(ell, i_tomo, j_tomo);
    }
    uint tid = omp_get_thread_num();
    integration_variable_Limber_ell[tid] = ell;
    integration_variable_Limber_i_tomo[tid] = i_tomo;
    integration_variable_Limber_j_tomo[tid] = j_tomo;
    integration_variable_Limber_linear[tid] = linear;
    double min = chi_min;
    double max = chi_max;
    return fac * gslIntegrateqag(Limber_kernel, min, max);
}

double Levin_power::Limber_kernel(double chi, void *p)
{
    uint tid = omp_get_thread_num();
    Levin_power *lp = static_cast<Levin_power *>(p);
    double weight_i_tomo = lp->kernels(chi, lp->integration_variable_Limber_i_tomo[tid]);
    double weight_j_tomo = lp->kernels(chi, lp->integration_variable_Limber_j_tomo[tid]);
    double k = (lp->integration_variable_Limber_ell[tid] + 0.5) / chi;
    double power = 0.0;
    double z = lp->z_of_chi(chi);
    if (lp->integration_variable_Limber_linear[tid])
    {
        power = lp->power_linear(z, k);
    }
    else
    {
        power = lp->power_nonlinear(z, k);
    }
    return weight_i_tomo * weight_j_tomo / gsl_pow_2(chi) * power;
}

double Levin_power::auxillary_weight(uint i_tomo, double k)
{
    if (k <= aux_kmin.at(i_tomo))
    {
        double slope = gsl_spline_eval_deriv(spline_aux_kernel.at(i_tomo), log(aux_kmin.at(i_tomo)), acc_aux_kernel.at(i_tomo)) / gsl_spline_eval(spline_aux_kernel.at(i_tomo), log(aux_kmin.at(i_tomo)), acc_aux_kernel.at(i_tomo));
        double result = abs(slope) * (log(k) - log(aux_kmin.at(i_tomo))) + log(abs(gsl_spline_eval(spline_aux_kernel.at(i_tomo), log(aux_kmin.at(i_tomo)), acc_aux_kernel.at(i_tomo))));
        return exp(-abs(result));
    }
    if (k >= aux_kmax.at(i_tomo))
    {
        return 0.0;
    }
    else
    {
        return gsl_spline_eval(spline_aux_kernel.at(i_tomo), log(k), acc_aux_kernel.at(i_tomo));
    }
}

double Levin_power::k_integration_kernel(double k, void *p)
{
    uint tid = omp_get_thread_num();
    k = exp(k);
    Levin_power *lp = static_cast<Levin_power *>(p);
    return k * lp->auxillary_weight(lp->integration_variable_i_tomo[tid], k) * lp->auxillary_weight(lp->integration_variable_j_tomo[tid], k);
}

double Levin_power::dlnP_dlnk(double k, double z)
{
    return gsl_spline2d_eval_deriv_x(spline_P_nl, log(k), z, acc_P_nl_k, acc_P_nl_z);
}

double Levin_power::d2P_d2k(double k, double z)
{
    return (gsl_spline2d_eval_deriv_xx(spline_P_nl, log(k), z, acc_P_nl_k, acc_P_nl_z) + gsl_pow_2(dlnP_dlnk(k, z)) - dlnP_dlnk(k, z)) * power_nonlinear(z, k) / gsl_pow_2(k);
}

double Levin_power::d2P_d2k_interp(double k, double z)
{
    return gsl_spline2d_eval(spline_d2P_d2k, log(k), z, acc_d2P_d2k_k, acc_d2P_d2k_z);
}

double Levin_power::d3P_d3k(double k, double z)
{
    return gsl_spline2d_eval_deriv_x(spline_d2P_d2k, log(k), z, acc_d2P_d2k_k, acc_d2P_d2k_z) * k;
}

double Levin_power::dlnkernels_dlnchi(double chi, uint i_tomo)
{
    return gsl_spline_eval_deriv(spline_Weight.at(i_tomo), chi, acc_Weight.at(i_tomo)) / kernels(chi, i_tomo) * chi;
}

double Levin_power::extended_limber_s(double k, double z)
{
    return dlnP_dlnk(k, z);
}

double Levin_power::extended_limber_p(double k, double z)
{
    return gsl_pow_2(k) * ((3.0 * d2P_d2k_interp(k, z) + k * d3P_d3k(k, z)) / (3.0 * power_nonlinear(z, k)));
}

double Levin_power::extended_Limber_kernel(double chi, void *p)
{
    chi = exp(chi);
    uint tid = omp_get_thread_num();
    Levin_power *lp = static_cast<Levin_power *>(p);
    double weight_i_tomo = lp->kernels(chi, lp->integration_variable_extended_Limber_i_tomo[tid]);
    double weight_j_tomo = lp->kernels(chi, lp->integration_variable_extended_Limber_j_tomo[tid]);
    double k = (lp->integration_variable_extended_Limber_ell[tid] + 0.5) / chi;
    double z = lp->z_of_chi(chi);
    double power = lp->power_nonlinear(z, k);
    double limber_part = weight_i_tomo * weight_j_tomo / chi * power;
    double dlnf_i = lp->dlnkernels_dlnchi(chi, lp->integration_variable_extended_Limber_i_tomo[tid]) - 0.5 / sqrt(chi);
    double dlnf_j = lp->dlnkernels_dlnchi(chi, lp->integration_variable_extended_Limber_j_tomo[tid]) - 0.5 / sqrt(chi);
    double correction = 0.5 / gsl_pow_2(lp->integration_variable_extended_Limber_ell[tid] + 0.5) * (dlnf_i * dlnf_j * lp->extended_limber_s(k, z) - lp->extended_limber_p(k, z));
    return limber_part * (1.0 + correction);
}

double Levin_power::extended_Limber(uint ell, uint i_tomo, uint j_tomo)
{
    uint tid = omp_get_thread_num();
    integration_variable_extended_Limber_ell[tid] = ell;
    integration_variable_extended_Limber_i_tomo[tid] = i_tomo;
    integration_variable_extended_Limber_j_tomo[tid] = j_tomo;
    double min = chi_min;
    double max = chi_max;
    return gslIntegratecquad(extended_Limber_kernel, log(min), log(max));
}

double Levin_power::C_ell_full(uint i_tomo, uint j_tomo)
{
    uint tid = omp_get_thread_num();
    integration_variable_i_tomo[tid] = i_tomo;
    integration_variable_j_tomo[tid] = j_tomo;
    double min = k_min; //(GSL_MIN(aux_kmin.at(i_tomo), aux_kmin.at(j_tomo)));
    double max = (GSL_MAX(aux_kmax.at(i_tomo), aux_kmax.at(j_tomo)));
    return 2.0 / M_PI * gslIntegrateqag(k_integration_kernel, log(min), log(max));
}

std::vector<double> Levin_power::all_C_ell(std::vector<uint> ell, bool linear)
{
    std::vector<double> result(ell.size() * n_total * n_total, 0.0);
    for (uint l = 0; l < ell.size(); l++)
    {
        set_auxillary_splines(ell.at(l), linear);
        double factor1 = factor[ell.at(l)];
#pragma omp parallel for
        for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
        {
            for (uint j_tomo = i_tomo; j_tomo < n_total; j_tomo++)
            {
                double facaux = 1.0;
                if (i_tomo >= number_counts)
                {
                    facaux *= factor1;
                }
                if (j_tomo >= number_counts)
                {
                    facaux *= factor1;
                }
                auto flat_idx = i_tomo * n_total + j_tomo;
                if ((ell.at(l) < ell_eLimber.at(i_tomo) && ell.at(l) < ell_eLimber.at(j_tomo)))
                {
                    result.at(l * n_total * n_total + flat_idx) = facaux * C_ell_full(i_tomo, j_tomo);
                    double aux;
                    if (ell.at(l) < ell_eLimber.at(j_tomo) && i_tomo == j_tomo)
                    {
                        aux = facaux * Limber(ell.at(l), i_tomo, j_tomo, linear);
                        double residual = (result.at(l * n_total * n_total + flat_idx) - aux) / aux;
                        if (abs(residual) <= eLimber_rel)
                        {
                            ell_eLimber.at(i_tomo) = ell.at(l);
                        }
                    }
                }
                else
                {
                    result.at(l * n_total * n_total + flat_idx) = facaux * Limber(ell.at(l), i_tomo, j_tomo, linear);
                }
            }
        }
    }
    return result;
}

std::tuple<result_Cl_type, result_Cl_type, result_Cl_type> Levin_power::compute_C_ells(std::vector<uint> ell)
{
    int n_tomo_A = number_counts;
    result_Cl_type Cl_AA;
    result_Cl_type Cl_BB;
    result_Cl_type Cl_AB;

    auto tmp_Cl = std::vector<double>(ell.size());

    auto result = all_C_ell(ell, false);

    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        for (uint j_tomo = i_tomo; j_tomo < n_total; j_tomo++)
        {
            auto flat_idx = i_tomo * n_total + j_tomo;
            for (uint l = 0; l < ell.size(); l++)
            {
                tmp_Cl.at(l) = result.at(l * n_total * n_total + flat_idx);
            }
            // Assign to auto and cross-correlation vectors
            if (i_tomo < n_tomo_A && j_tomo < n_tomo_A)
            {
                Cl_AA.push_back(tmp_Cl);
            }
            else if (i_tomo >= n_tomo_A && j_tomo >= n_tomo_A)
            {
                Cl_BB.push_back(tmp_Cl);
            }
            else
            {
                Cl_AB.push_back(tmp_Cl);
            }
        }
    }

    return std::make_tuple(Cl_AA, Cl_AB, Cl_BB);
}

double Levin_power::gslIntegrateqag(double (*fc)(double, void *), double a, double b)
{

    double tiny = 0.0;
    double tol = 1.0e-3;
    gsl_function gf;
    gf.function = fc;
    gf.params = this;
    double e, y;
    const uint n = 128;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(n);
    gsl_integration_qag(&gf, a, b, tiny, tol, n, 1, w, &y, &e);
    gsl_integration_workspace_free(w);
    return y;
}

double Levin_power::gslIntegrateqng(double (*fc)(double, void *), double a, double b)
{

    double tiny = 0.0;
    double tol = 1.0e-4;
    gsl_function gf;
    gf.function = fc;
    gf.params = this;
    double e, y;
    size_t n;
    gsl_integration_qng(&gf, a, b, tiny, tol, &y, &e, &n);
    return y;
}

double Levin_power::gslIntegratecquad(double (*fc)(double, void *), double a, double b)
{
    double tiny = 0.0;
    double tol = 1.0e-3;
    gsl_function gf;
    gf.function = fc;
    gf.params = this;
    double e, y;
    const uint n = 64;
    gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(n);
    gsl_integration_cquad(&gf, a, b, tiny, tol, w, &y, &e, NULL);
    gsl_integration_cquad_workspace_free(w);
    return y;
}