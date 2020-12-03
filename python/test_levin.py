import levinpower

import numpy as np
import pyccl as ccl

if __name__ == "__main__":
    pk = np.load("data/pk.npz")
    kernels = np.load("data/kernels.npz")
    background = np.load("data/background.npz")

    print("pk", [k for k in pk.keys()])
    print("kernels", [k for k in kernels.keys()])
    print("background", [k for k in background.keys()])
    
    number_count = kernels["kernels_cl"].shape[0]
    lp = levinpower.LevinPower(number_count, 
                          background["z"], background["chi"], 
                          kernels["chi_cl"], np.concatenate((kernels["kernels_cl"].T, kernels["kernels_sh"].T), axis=1), 
                          pk["k"], pk["z"], pk["pk_lin"].flatten(), pk["pk_nl"].flatten())


    ell  = np.arange(2) + 2
    ell = np.arange(2, 502, 100) 
    ell = np.unique(np.geomspace(2, 200, 4).astype(int))

    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)

    kernel_overlap_gg, kernel_overlap_gs, kernel_overlap_ss = lp.kernel_overlap

    print(len(Cl_gg), len(Cl_gs), len(Cl_ss))
    # print(Cl_gg)

    # Cl = lp.all_C_ell(ell, True)
    # Cl = np.array(Cl).reshape(len(ell),-1)

    # benchmark = np.loadtxt("mains/test.txt")[:,1::2]

    # print(Cl.shape, benchmark.shape)
    # assert np.allclose(Cl, benchmark[:Cl.shape[0]])

    par = {'Omega_m': 0.3156,
           'Omega_b': 0.0492,
           'w0': -1.0,
           'h': 0.6727,
           'A_s': 2.12107E-9,
           'n_s': 0.9645}
    cosmo = ccl.Cosmology(Omega_c=par['Omega_m']-par['Omega_b'],
                          Omega_b=par['Omega_b'],
                          h=par['h'], n_s=par['n_s'],
                          A_s=par['A_s'], w0=par['w0'])
    a = 1./(1+pk['z'][::-1])
    cosmo._set_linear_power_from_arrays(a_array=a,
                                        k_array=pk['k'],
                                        pk_array=pk['pk_lin'][::-1][:])
    cosmo._set_nonlin_power_from_arrays(a_array=a,
                                        k_array=pk['k'],
                                        pk_array=pk['pk_nl'][::-1][:])

    a_g = 1./(1+kernels['z_cl'][::-1])
    t_g = []
    for k in kernels['kernels_cl']:
        t = ccl.Tracer()
        t.add_tracer(cosmo, (kernels['chi_cl'], k))
        t_g.append(t)
    t_s = []
    for k in kernels['kernels_sh']:
        t = ccl.Tracer()
        t.add_tracer(cosmo, kernel=(kernels['chi_sh'], k), der_bessel=-1, der_angles=2)
        t_s.append(t)

    ccl_cls_gg = []
    ccl_cls_gs = []
    ccl_cls_ss = []
    for i1, t1 in enumerate(t_g):
        for t2 in t_g[i1:]:
            ccl_cls_gg.append(ccl.angular_cl(cosmo, t1, t2, ell))
        for t2 in t_s:
            ccl_cls_gs.append(ccl.angular_cl(cosmo, t1, t2, ell))
    for i1, t1 in enumerate(t_s):
        for t2 in t_s[i1:]:
            ccl_cls_ss.append(ccl.angular_cl(cosmo, t1, t2, ell))

    plot = True
    if plot:
        import matplotlib.pyplot as plt

        n_tomo_A = kernels["kernels_cl"].shape[0]
        n_tomo_B = kernels["kernels_sh"].shape[0]
        n_total = n_tomo_A + n_tomo_B


        fig, ax = plt.subplots(n_total, n_total, sharex=True, figsize=(10, 10))
        fig.subplots_adjust(wspace=0, hspace=0)

        for i in range(n_total):
            for j in range(n_total):
                if j < i:
                    ax[i,j].axis('off')
                    continue

                ax[i,j].set_facecolor("lightgrey")

                if i < n_tomo_A and j < n_tomo_A:
                    cl = Cl_gg
                    cl_ccl = ccl_cls_gg
                    flat_idx = (i*(2*n_tomo_A - i + 1))//2 + j-i
                    if kernel_overlap_gg[flat_idx]:
                        ax[i,j].set_facecolor("pink")
                    print("gg ", end="")
                elif i < n_tomo_A and j >= n_tomo_A:
                    cl = Cl_gs
                    cl_ccl = ccl_cls_gs
                    flat_idx = i*n_tomo_B + (j-n_tomo_A)
                    if kernel_overlap_gs[flat_idx]:
                        ax[i,j].set_facecolor("plum")
                    print("gs ", end="")
                else:
                    cl = Cl_ss
                    cl_ccl = ccl_cls_ss
                    flat_idx = ((i-n_tomo_A)*(2*n_tomo_B - (i-n_tomo_A) + 1))//2 + (j-i)
                    if kernel_overlap_ss[flat_idx]:
                        ax[i,j].set_facecolor("lightskyblue")
                    print("ss ", end="")

                print("i: ", i, " j: ", j, " flat_idx: ", flat_idx)

                ax[i,j].semilogx(ell, cl[flat_idx], c="C0")
                ax[i,j].semilogx(ell, cl_ccl[flat_idx], c="C1")

        plt.show()




