import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.special import spherical_jn
import scipy.integrate as integrate
from joblib import Parallel, delayed
import multiprocessing

min_interval = 1.0e-2
tol_rel = 1.0e-6      # desired relative accuracy
epsilon = 1.0e-12  # limit for the absolute error (due to machine accuracy)
tol_abs = epsilon
N_interp = 150  # Number of interpolation points used for the k integration

num_cores = multiprocessing.cpu_count()
data_background = np.load("background.npz")
lst = data_background.files
redshift = data_background['z']
comoving_distance = data_background['chi']
chi_of_z = interp1d(redshift, comoving_distance)
z_of_chi = interp1d(comoving_distance, redshift)


data_kernels = np.load("kernels.npz")
lst = data_kernels.files
comoving_distance_cl = data_kernels['chi_cl']
comoving_distance_sh = data_kernels['chi_sh']
kernels_cl = data_kernels['kernels_cl']
n_number_counts = len(kernels_cl[:, 0])
kernel = []
kernels_sh = data_kernels['kernels_sh']
n_cosmic_shear = len(kernels_sh[:, 0])
kernel_maximum = np.zeros(n_number_counts+n_cosmic_shear)
for i in range(n_number_counts+n_cosmic_shear):
    if(i < n_number_counts):
        kernel.append(
            interp1d(comoving_distance_cl, kernels_cl[i, :], kind='cubic'))
        kernel_maximum[i] = comoving_distance_cl[np.where(
            kernels_cl[i, :] == max(kernels_cl[i, :]))]
    else:
        kernel.append(
            interp1d(comoving_distance_sh, kernels_sh[i-n_number_counts, :], kind='cubic'))
        kernel_maximum[i] = comoving_distance_cl[np.where(
            kernels_sh[i-n_number_counts, :] == max(kernels_sh[i-n_number_counts, :]))]
chi_min = (comoving_distance_cl[0])+1.0
chi_max = (comoving_distance_cl[-1])-1.0
data_power = np.load("pk.npz")
lst = data_power.files
k = data_power['k']
z = data_power['z']
pk_nl = data_power['pk_nl']
pk = data_power['pk_lin']
linear_power_spectrum_spline = RectBivariateSpline(z, np.log(k), np.log(pk))
nonlinear_power_spectrum_spline = RectBivariateSpline(
    z, np.log(k), np.log(pk_nl))
k_min = k[0]
k_max = k[-1]


def linear_power_spectrum(z, k):
    return np.exp(linear_power_spectrum_spline(z, np.log(k))[0, 0])


def nonlinear_power_spectrum(z, k):
    return np.exp(nonlinear_power_spectrum_spline(z, np.log(k))[0, 0])


def w(chi, k, ell, i):
    if(i == 0):
        return spherical_jn(ell, chi*k)
    if(i == 1):
        return spherical_jn(ell-1, chi*k)


def A_matrix(i, j, chi, k, ell):
    if (i == 0 and j == 0):
        return -(ell+1.0)/chi
    if(i*j == 1):
        return (ell-1.0)/chi
    if(i < j):
        return k
    else:
        return -k


def setNodes(A, B, col):
    n = col
    if(int(col) % 2 != 0):
        n = col+1
    x_j = np.zeros(int(n))
    for j in range(int(n)):
        x_j[j] = A+j*(B-A)/(n-1)
    return x_j


def basis_function(A, B, x, m):
    if(m == 0):
        return 1.0
    u = ((x-(A+B)/2)/(B-A))**m
    return u


def basis_function_prime(A, B, x, m):
    if(m == 0):
        return 0.0
    if(m == 1):
        return 1.0/(B-A)
    uprime = m / (B - A) * ((x - (A + B) / 2.) / (B - A))**(m - 1)
    return uprime


def F_linear(chi, i_tomo, k):
    z = z_of_chi(chi)
    return np.sqrt(linear_power_spectrum(z, k))*kernel[i_tomo](chi)


def F_nonlinear(chi, i_tomo, k):
    z = z_of_chi(chi)
    return np.sqrt(nonlinear_power_spectrum(z, k))*kernel[i_tomo](chi)


def solve_LSE_linear(A, B, col, x_j, d, i_tomo, k, ell):
    n = col
    if(int(col) % 2 != 0):
        n = col+1
    n = int(n)
    F_stacked = np.zeros((d*n))
    for j in range(n):
        F_stacked[j] = F_linear(x_j[j], i_tomo, k)
    matrix_G = np.zeros((d*n, d*n))
    for i in range(d):
        for j in range(n):
            for q in range(d):
                for m in range(n):
                    LSE_coeff = A_matrix(q, i, x_j[j], k, ell) * \
                        basis_function(A, B, x_j[j], m)
                    if (q == i):
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m)
                    matrix_G[i * n + j][q * n + m] = LSE_coeff
    return np.linalg.solve(matrix_G, F_stacked)


def solve_LSE_nonlinear(A, B, col, x_j, d, i_tomo, k, ell):
    n = col
    if(int(col) % 2 != 0):
        n = col+1
    n = int(n)
    F_stacked = np.zeros(d*n)
    for j in range(n):
        F_stacked[j] = F_nonlinear(x_j[j], i_tomo, k)
    matrix_G = np.zeros(d*n, d*n)
    for i in range(d):
        for j in range(n):
            for q in range(d):
                for m in range(n):
                    LSE_coeff = A_matrix(q, i, x_j[j], k, ell) * \
                        basis_function(A, B, x_j[j], m)
                    if (q == i):
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m)
                    matrix_G[i * n + j][q * n + m] = LSE_coeff
    inverse = np.linalg.pinv(matrix_G)
    # return np.linalg.solve(matrix_G, F_stacked)
    return(np.dot(inverse, F_stacked))


def p(A, B, i, x, col, c):
    n = col
    if(int(col) % 2 != 0):
        n = col+1
    n = int(n)
    res = 0.0
    for m in range(n):
        res += c[i*n+m]*basis_function(A, B, x, m)
    return res


def integrate_linear(A, B, col, d, i_tomo, k, ell):
    res = 0.0
    x_j = setNodes(A, B, col)
    c = solve_LSE_linear(A, B, col, x_j, d, i_tomo, k, ell)
    for i in range(d):
        res += p(A, B, i, B, col, c)*w(B, k, ell, i) - \
            p(A, B, i, A, col, c)*w(A, k, ell, i)
    return res


def integrate_nonlinear(A, B, col, d, i_tomo, k, ell):
    res = 0.0
    x_j = setNodes(A, B, col)
    c = solve_LSE_nonlinear(A, B, col, x_j, d, i_tomo, k, ell)
    for i in range(d):
        res += p(A, B, i, B, col, c)*w(B, k, ell, i) - \
            p(A, B, i, A, col, c)*w(A, k, ell, i)
    return res


def levin_interate_linear(A, B, col, d, i_tomo, k, ell, smax, verbose):
    intermediate_results = []
    if (B - A < min_interval):
        return 0
    borders = [A, B]
    x_sub = borders
    I_half = integrate_linear(A, B, col / 2, d, i_tomo, k, ell)
    I_full = integrate_linear(A, B, col, d, i_tomo, k, ell)
    sub = 1  # number of subintervals
    previous = I_half
    approximations = []
    # integral values in subintervals
    approximations.append(I_full)
    result = I_full
    error_estimates = []
    # (absolute) error estimates for subintervals
    error_estimates.append(abs(I_full - I_half))
    while (sub <= smax+1):
        result = 0.0
        for i in range(len(approximations)):
            result += approximations[i]
        if (verbose):  # print borders, approximations and error estimates for all subintervals
            print("estimate: ", result)
            print("subintervals: ", sub)
            for i in range(len(approximations)):
                print("[", x_sub[i], ",", x_sub[i + 1], "]: ",
                      approximations[i], " (", error_estimates[i], ")")
        intermediate_results.append(result)
        # check if relative accuracy has been reached (or total change in result compared to previous step is negligible)
        if (abs(result - previous) <= max(tol_rel * abs(result), tol_abs)):
            if (verbose):
                print("converged!")
            return result
        # if not, set up for next step
        previous = result
        sub += 1
        i = 1
        while (True):
            # identify the subinterval with the largest estimated error
            i = error_estimates.index(max(error_estimates)) + 1
            # this is the case if all subintervals are too small to be bisected
            if (error_estimates[i - 1] < 0):
                if (verbose):
                    print("subintervals too narrow for further bisection!")
                return 0
            if (x_sub[i] - x_sub[i - 1] > min_interval):
                break                          # check if it is "large enough" to be bisected
            error_estimates[i - 1] = -1.0
        # divide subinterval
        x_sub.insert(i, (x_sub[i - 1] + x_sub[i]) / 2.)
        I_half = integrate_linear(
            x_sub[i-1], x_sub[i], col / 2, d, i_tomo, k, ell)
        I_full = integrate_linear(x_sub[i-1], x_sub[i], col, d, i_tomo, k, ell)
        approximations[i - 1] = I_full
        error_estimates[i - 1] = abs(I_full - I_half)
        I_half = integrate_linear(
            x_sub[i], x_sub[i+1], col / 2, d, i_tomo, k, ell)
        I_full = integrate_linear(x_sub[i], x_sub[i+1], col, d, i_tomo, k, ell)
        approximations.insert(i, I_full)
        error_estimates.insert(i, abs(I_full - I_half))
    if (verbose):
        print("maximum number of subintervals reached!")
    return result


def levin_interate_nonlinear(A, B, col, d, i_tomo, k, ell, smax, verbose):
    intermediate_results = []
    if (B - A < min_interval):
        return 0
    borders = [A, B]
    x_sub = borders
    I_half = integrate_nonlinear(A, B, col / 2, d, i_tomo, k, ell)
    I_full = integrate_nonlinear(A, B, col, d, i_tomo, k, ell)
    sub = 1  # number of subintervals
    previous = I_half
    approximations = []
    # integral values in subintervals
    approximations.append(I_full)
    result = I_full
    error_estimates = []
    # (absolute) error estimates for subintervals
    error_estimates.append(abs(I_full - I_half))
    while (sub <= smax+1):
        result = 0.0
        for i in range(len(approximations)):
            result += approximations[i]
        if (verbose):  # print borders, approximations and error estimates for all subintervals
            print("estimate: ", result)
            print("subintervals: ", sub)
            for i in range(len(approximations)):
                print("[", x_sub[i], ",", x_sub[i + 1], "]: ",
                      approximations[i], " (", error_estimates[i], ")")
        intermediate_results.append(result)
        # check if relative accuracy has been reached (or total change in result compared to previous step is negligible)
        if (abs(result - previous) <= max(tol_rel * abs(result), tol_abs)):
            if (verbose):
                print("converged!")
            return result
        # if not, set up for next step
        previous = result
        sub += 1
        i = 1
        while (True):
            # identify the subinterval with the largest estimated error
            i = error_estimates.index(max(error_estimates)) + 1
            # this is the case if all subintervals are too small to be bisected
            if (error_estimates[i - 1] < 0):
                if (verbose):
                    print("subintervals too narrow for further bisection!")
                return 0
            if (x_sub[i] - x_sub[i - 1] > min_interval):
                break                          # check if it is "large enough" to be bisected
            error_estimates[i - 1] = -1.0
        # divide subinterval
        x_sub.insert(i, (x_sub[i - 1] + x_sub[i]) / 2.)
        I_half = integrate_nonlinear(
            x_sub[i-1], x_sub[i], col / 2, d, i_tomo, k, ell)
        I_full = integrate_nonlinear(
            x_sub[i-1], x_sub[i], col, d, i_tomo, k, ell)
        approximations[i - 1] = I_full
        error_estimates[i - 1] = abs(I_full - I_half)
        I_half = integrate_nonlinear(
            x_sub[i], x_sub[i+1], col / 2, d, i_tomo, k, ell)
        I_full = integrate_nonlinear(
            x_sub[i], x_sub[i+1], col, d, i_tomo, k, ell)
        approximations.insert(i, I_full)
        error_estimates.insert(i, abs(I_full - I_half))
    if (verbose):
        print("maximum number of subintervals reached!")
    return result


def levin_integrate_bessel(k, ell, i_tomo, linear=True):
    n_col = 8
    n_sub = 12
    if(linear):
        return levin_interate_linear(chi_min, chi_max, n_col, 2, i_tomo, k, ell, n_sub, False)*k
    else:
        return levin_interate_nonlinear(chi_min, chi_max, n_col, 2, i_tomo, k, ell, n_sub, False)*k


def exact_integral_bessel(k, ell, i_tomo, linear=True):
    N_calls = 1e4
    chi = np.linspace(chi_min, chi_max, int(N_calls))
    integral = 0.0
    if(linear):
        for i in range(int(N_calls)-1):
            integral += spherical_jn(ell, k*chi[i]) * \
                (chi[i+1]-chi[i])*F_linear(chi[i], i_tomo, k)
    else:
        for i in range(int(N_calls)-1):
            integral += spherical_jn(ell, k*chi[i]) * \
                (chi[i+1]-chi[i])*F_linear(chi[i], i_tomo, k)
    return integral


def Limber(ell, i_tomo, j_tomo, linear=False):
    if(linear):
        return integrate.quad(lambda x: F_linear(x, i_tomo, (ell+0.5)/x)*F_linear(x, j_tomo, (ell+0.5)/x)/x**2.0, chi_min, chi_max)[0]
    else:
        return integrate.quad(lambda x: F_nonlinear(x, i_tomo, (ell+0.5)/x)*F_nonlinear(x, j_tomo, (ell+0.5)/x)/x**2.0, chi_min, chi_max)[0]


def C_ell_full_expression(ell, i_tomo, j_tomo, linear=False):
    kmin_i_tomo = max(0.1*(ell+0.5)/kernel_maximum[i_tomo], k_min)
    kmax_i_tomo = min(10.0*(ell+0.5)/kernel_maximum[i_tomo], k_max)
    kmin_j_tomo = max(0.1*(ell+0.5)/kernel_maximum[j_tomo], k_min)
    kmax_j_tomo = min(10.0*(ell+0.5)/kernel_maximum[j_tomo], k_max)
    if(i_tomo >= n_number_counts):
        kmax_i_tomo = 1.0
    if(j_tomo >= n_number_counts):
        kmax_j_tomo = 1.0
    k_interp_i_tomo = np.linspace(
        np.log(kmin_i_tomo), np.log(kmax_i_tomo), N_interp)
    k_interp_j_tomo = np.linspace(
        np.log(kmin_j_tomo), np.log(kmax_j_tomo), N_interp)
    I_bessel_i_tomo = np.zeros(N_interp)
    I_bessel_j_tomo = np.zeros(N_interp)
    if(i_tomo == j_tomo):
        I_bessel_i_tomo = Parallel(n_jobs=num_cores)(delayed(levin_integrate_bessel)(np.exp(
            k_interp_i_tomo[i]), int(ell), i_tomo, linear) for i in range(N_interp))
        aux_interp = interp1d(
            k_interp_i_tomo, I_bessel_i_tomo, kind='cubic')
        result = 2.0/np.pi * integrate.quad(lambda x: aux_interp(np.log(x)) **
                                            2.0, kmin_i_tomo, kmax_i_tomo)[0]
        return result
    else:
        I_bessel_i_tomo = Parallel(n_jobs=num_cores)(delayed(levin_integrate_bessel)(np.exp(
            k_interp_i_tomo[i]), int(ell), i_tomo, linear) for i in range(N_interp))
        I_bessel_j_tomo = Parallel(n_jobs=num_cores)(delayed(levin_integrate_bessel)(np.exp(
            k_interp_j_tomo[i]), int(ell), j_tomo, linear) for i in range(N_interp))
        aux_interp_i = interp1d(
            k_interp_i_tomo, I_bessel_i_tomo, kind='cubic')
        aux_interp_j = interp1d(
            k_interp_j_tomo, I_bessel_j_tomo, kind='cubic')

        def aux_weight_i_tomo(x):
            if(x < k_interp_i_tomo[0] or x > k_interp_i_tomo[-1]):
                return 0.0
            else:
                return aux_interp_i(x)

        def aux_weight_j_tomo(x):
            if(x < k_interp_j_tomo[0] or x > k_interp_j_tomo[-1]):
                return 0.0
            else:
                return aux_interp_j(x)
        result = 2.0/np.pi * integrate.quad(lambda x: aux_weight_i_tomo(np.log(x))
                                            * aux_weight_j_tomo(np.log(x)), np.exp(min(k_interp_i_tomo[0], k_interp_j_tomo[0])), np.exp(max(k_interp_i_tomo[-1], k_interp_j_tomo[-1])))[0]
        return result


def all_C_ell_limber(ell_list, linear=False):
    ntotal = 3  # n_cosmic_shear + n_number_counts
    result = np.zeros((len(ell_list), ntotal, ntotal))
    for i in range(len(ell_list)):
        ell = int(ell_list[i])
        for i_tomo in range(ntotal):
            for j_tomo in range(i_tomo, ntotal):
                result[i][i_tomo][j_tomo] = Limber(
                    ell, i_tomo, j_tomo, linear)
    return result


def all_C_ell_full(ell_list, linear=False):
    Limber_tolerance = 1e-3
    min_ell_check_Limber = 50
    ntotal = 3  # n_cosmic_shear + n_number_counts
    result = np.zeros((len(ell_list), ntotal, ntotal))
    use_limber = np.zeros(ntotal)
    for i in range(len(ell_list)):
        ell = int(ell_list[i])
        aux_kernel = []
        aux_kmax = []
        aux_kmin = []
        for i_tomo in range(ntotal):
            if(use_limber[i_tomo] == 0):
                kmin_i_tomo = max(0.1*(ell+0.5)/kernel_maximum[i_tomo], k_min)
                kmax_i_tomo = min(10.0*(ell+0.5)/kernel_maximum[i_tomo], k_max)
                if(i_tomo >= n_number_counts):
                    kmax_i_tomo = 1.0
                k_interp_i_tomo = np.linspace(
                    np.log(kmin_i_tomo), np.log(kmax_i_tomo), N_interp)
                I_bessel_i_tomo = Parallel(n_jobs=num_cores)(delayed(levin_integrate_bessel)(np.exp(
                    k_interp_i_tomo[i]), int(ell), i_tomo, linear) for i in range(N_interp))
                aux_kernel.append(
                    interp1d(k_interp_i_tomo, I_bessel_i_tomo, kind='cubic'))
                aux_kmax.append(k_interp_i_tomo[-1])
                aux_kmin.append(k_interp_i_tomo[0])
            else:
                aux_kernel.append(0.0)
                aux_kmax.append(k_interp_i_tomo[-1])
                aux_kmin.append(k_interp_i_tomo[0])

        def aux_weight(x, i_tomo):
            if(x < aux_kmin[i_tomo] or x > aux_kmax[i_tomo]):
                return 0.0
            else:
                return aux_kernel[i_tomo](x)

        def final_integral(ell, a):
            i_tomo = int(a/ntotal)
            j_tomo = a - i_tomo*ntotal
            aux_result = 0.0
            if(i_tomo <= j_tomo):
                if(use_limber[i_tomo] == 0):
                    aux_result = 2.0/np.pi * integrate.quad(lambda x: aux_weight(np.log(x), i_tomo)
                                                            * aux_weight(np.log(x), j_tomo), np.exp(min(aux_kmin[i_tomo], aux_kmin[j_tomo])), np.exp(max(aux_kmax[i_tomo], aux_kmax[j_tomo])))[0]
                else:
                    aux_result = Limber(
                        ell, i_tomo, j_tomo, linear)
            return aux_result

        def integral_limber(ell, a):
            i_tomo = int(a/(ntotal))
            j_tomo = a - i_tomo*ntotal
            aux_result = 0.0
            if(i_tomo <= j_tomo):
                aux_result = Limber(
                    ell, i_tomo, j_tomo, linear)
            return aux_result

        aux_full = np.zeros(ntotal*ntotal)
        aux_Limber = np.zeros(ntotal * ntotal)

        aux_full = Parallel(n_jobs=num_cores)(delayed(final_integral)(ell, i)
                                              for i in range(ntotal*ntotal))
        for i_tomo in range(ntotal):
            for j_tomo in range(i_tomo, ntotal):
                result[i][i_tomo][j_tomo] = aux_full[i_tomo*ntotal + j_tomo]
        if(ell > min_ell_check_Limber):
            aux_Limber = Parallel(n_jobs=num_cores)(delayed(integral_limber)(ell, i)
                                                    for i in range(ntotal*ntotal))
            for i_tomo in range(ntotal):
                test_aux = ntotal - i_tomo
                for j_tomo in range(i_tomo, ntotal):
                    if(np.abs(aux_Limber[i_tomo*ntotal + j_tomo] - aux_full[i_tomo*ntotal + j_tomo]) < Limber_tolerance):
                        test_aux -= 1
                if(test_aux == 0):
                    use_limber[i_tomo] = 1
    return result


N = 30
# x=np.exp(np.linspace(np.log(1e-4), np.log(1e-2), N))
x = np.linspace(2, 90, N)
# y = []
# y1 = []
ntotal = 3  # n_cosmic_shear + n_number_counts
spectra = all_C_ell_full(x, True)
spectra_limber = all_C_ell_limber(x, True)

# Plotting

fontsi = 8
fontsi2 = 8
plt.tick_params(labelsize=fontsi)
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams['xtick.labelsize'] = '8'
plt.rcParams['ytick.labelsize'] = '8'
fig, ax = plt.subplots(ntotal, ntotal)
max_yticks = 2
max_xticks = 2
# yloc = plt.MaxNLocator(max_yticks)
# xloc = plt.MaxNLocator(max_yticks)
for i in range(ntotal):
    for j in range(i):
        ax[i, j].axis('off')

# for i in range(ntotal-1):
#    a = i + 1
#    for j in range(a, ntotal):
#        ax[i, j].set_xticklabels([])


for i in range(ntotal):
    for j in range(i, ntotal):
        ax[i, j].set_yscale('log')
        ax[i, j].set_xscale('log')
        y = np.zeros(N)
        y1 = np.zeros(N)
        for a in range(N):
            y[a] = x[a]*(x[a]+1)*spectra[a][i][j]
            y1[a] = x[a]*(x[a]+1)*spectra_limber[a][i][j]
        ax[i, j].plot(x, y, ls="-", color="blue", lw=1)
        ax[i, j].plot(x, y1, ls="-", color="red", lw=1)

        # ax[i, j].plot(x, y1 , label =r"\mathrm{Limber}", ls="-", color="blue")
        # ax[i, j].set_xscale('log')


for i in range(ntotal):
    for j in range(ntotal):
        if(i != j):
            ax[i, j].set_xticklabels([])
#            ax[i, j].set_yticklabels([])
        else:
            ax[i, j].set_xlabel(r"$\ell$", fontsize=fontsi)
            ax[i, j].set_ylabel(
                r"$\ell(\ell+1) C_\ell$", fontsize=fontsi)

plt.subplots_adjust(wspace=0.0)
plt.subplots_adjust(hspace=0.0)
# leg = ax[0, 0].legend(fancybox=True, loc='upper right',
# fontsize=fontsi, frameon=False)


plt.tight_layout()

plt.savefig('Spectra_full.pdf')

# plt.plot(x, y1, ls="--")

# plt.yscale('log')
# plt.xscale('log')
# plt.show()
