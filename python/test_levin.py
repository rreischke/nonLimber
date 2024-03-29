import levinpower

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pk = np.load("./../data/pk.npz")
    kernels = np.load("./../data/kernels.npz")
    background = np.load("./../data/background.npz")

    #print("pk", [k for k in pk.keys()])
    #print("kernels", [k for k in kernels.keys()])
    #print("background", [k for k in background.keys()])
    
    number_count = kernels["kernels_cl"].shape[0]
    ell  = np.arange(2,2001,1)

    lp = levinpower.LevinPower(ell, number_count, 
                          background["z"], background["chi"], 
                          kernels["chi_cl"], np.concatenate((kernels["kernels_cl"].T, kernels["kernels_sh"].T), axis=1), 
                          pk["k"], pk["z"], pk["pk_lin"].flatten(), pk["pk_nl"].flatten())


    
    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells()

    Cl_gg = np.array(Cl_gg)
    Cl_gs = np.array(Cl_gs)
    Cl_ss = np.array(Cl_ss)

    print(Cl_gg.shape)
    #print(Cl_gg)

    # plt.plot(ell, Cl_gg[0])
    # plt.plot(ell, Cl_gg[1])
    # plt.plot(ell, Cl_gg[2])
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    Cl = lp.all_C_ell(ell, True)
    Cl = np.array(Cl).reshape(len(ell),-1)

    benchmark = np.loadtxt("../mains/test.txt")[:,1::2]

   # print(Cl.shape, benchmark.shape)
    assert np.allclose(Cl, benchmark[:Cl.shape[0]])

