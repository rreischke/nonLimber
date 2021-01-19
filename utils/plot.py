import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.special import spherical_jn
import scipy.integrate as integrate
from joblib import Parallel, delayed
import multiprocessing

ngg = 10
ntotal_spec = 15
spectra = np.loadtxt("./../mains/test2.txt")
data_gg = np.load("benchmarks_nl_clgg.npz")
data_gs = np.load("benchmarks_nl_clgs.npz")
data_ss = np.load("benchmarks_nl_clss.npz")

# for i in range(len(data_gg['ls'])):
#    print(data['ls'][i])
for i in range (len(data_gg['ls'][:])):
    print(data_gg['ls'][i,],end=' ')
    for j in range(55):
        print(data_gg['cls'][j, i],end=' ')
    print(' ')
N = len(spectra[:, 0])
x = spectra[:, 0]
ntotal = 5
fontsi = 8
fontsi2 = 8
plt.tick_params(labelsize=fontsi)
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams['xtick.labelsize'] = '8'
plt.rcParams['ytick.labelsize'] = '8'
fig, ax = plt.subplots(ntotal, ntotal)
max_yticks = 4
max_xticks = 4
yloc = plt.MaxNLocator(max_yticks)
xloc = plt.MaxNLocator(max_yticks)
for i in range(ntotal):
    for j in range(ntotal):
        if(j<=i):
            ax[i, j].set_xticklabels([])
        if(j>i):
            ax[i, j].axis('off')
flt_idx = 0
for i in range(0, ntotal):
    for j in range(0, i+1):
        #ax[i, j].set_yscale('log')
        ax[i, j].set_xscale('log')
        y = np.zeros(N)
        y1 = np.zeros(N)
        #print(i,j,flt_idx)
        #print(i, j, 2*(i)*ntotal_spec + 2*(j) + 1)
        for a in range(N):
            #y[a] = x[a]*(x[a]+1)*spectra[a,2*i*ntotal_spec + 2*j + 1]
            #y1[a] = x[a]*(x[a]+1)*spectra[a,2*i*ntotal_spec + 2*j + 2]
            y[a] = spectra[a, flt_idx + 1]
            y1[a] = data_gg['cls'][flt_idx, a]
            #y1[a] = spectra[a, 2*(i)*ntotal_spec + 2*(j) + 2]
        #ax[i-5, j-10].plot(x, 100.0*(y/y1 - 1.0), ls="-", color="blue", lw=1)
        ax[i, j].plot(x, y, ls="-", color="blue", lw=1)
        flt_idx += 1

       # ax[i, j-10].plot(x, y1, ls="--", color="red", lw=1)
        #ax[i, j].plot(x, y1, ls="-", color="blue", lw=1)
        #ax[i, j].plot(x, y1, ls="--", color="red", lw=2)
        #ax[i, j].set_xlim(x[0],x[len(x)-1])
        # ax[i, j].plot(x, y1 , label =r"\mathrm{Limber}", ls="-", color="blue")
        # ax[i, j].set_xscale('log')



plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)
# leg = ax[0, 0].legend(fancybox=True, loc='upper right',
# fontsize=fontsi, frameon=False)

# plt.tight_layout()

plt.savefig('Spectra_full_sh.pdf')
