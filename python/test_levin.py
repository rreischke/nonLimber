import levinpower

import numpy as np

if __name__ == "__main__":
    pk = np.load("./../data/pk.npz")
    kernels = np.load("./../data/kernels.npz")
    background = np.load("./../data/background.npz")

    print("pk", [k for k in pk.keys()])
    print("kernels", [k for k in kernels.keys()])
    print("background", [k for k in background.keys()])
    
    number_count = kernels["kernels_cl"].shape[0]
    lp = levinpower.LevinPower(number_count, 
                          background["z"], background["chi"], 
                          kernels["chi_cl"], np.concatenate((kernels["kernels_cl"].T, kernels["kernels_sh"].T), axis=1), 
                          pk["k"], pk["z"], pk["pk_lin"].flatten(), pk["pk_nl"].flatten())


    ell  = np.arange(2) + 2

    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)

    print(len(Cl_gg), len(Cl_gs), len(Cl_ss))
    print(Cl_gg)

    Cl = lp.all_C_ell(ell, True)
    Cl = np.array(Cl).reshape(len(ell),-1)

    benchmark = np.loadtxt("mains/test.txt")[:,1::2]

    print(Cl.shape, benchmark.shape)
    assert np.allclose(Cl, benchmark[:Cl.shape[0]])

