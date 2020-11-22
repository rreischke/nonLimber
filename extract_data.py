import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.special import spherical_jn
import scipy.integrate as integrate
from joblib import Parallel, delayed
import multiprocessing

data_background = np.load("background.npz")
lst = data_background.files
redshift = data_background['z']
comoving_distance = data_background['chi']
chi_of_z = interp1d(redshift, comoving_distance)
z_of_chi = interp1d(comoving_distance, redshift)

#for j in range(len(comoving_distance)):
#    print(redshift[j],comoving_distance[j])


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
print (len(comoving_distance_sh),len(comoving_distance_cl))
for j in range(len(comoving_distance_cl)):
    #print(comoving_distance_cl[j], end=' ')
    for i in range(n_number_counts):
        result = 0.0
        if(i < n_number_counts):
            result = kernels_cl[i, j] 
        else:
            result = kernels_sh[i-n_number_counts, j]
     #   print(result, end=' ')
   # print()

data_power = np.load("pk.npz")
lst = data_power.files
k = data_power['k']

#for i in range (len(k)):
z = data_power['z']
#print(len(k), len(z))
pk_nl = data_power['pk_nl']
pk = data_power['pk_lin']
#print(len(pk))
#for i in range(len(z)):
#    for j in range(len(k)):
#        print(z[i],k[j],pk[i][j],pk_nl[i][j])

linear_power_spectrum_spline = RectBivariateSpline(z, np.log(k), np.log(pk))
nonlinear_power_spectrum_spline = RectBivariateSpline(
    z, np.log(k), np.log(pk_nl))
k_min = k[0]
k_max = k[-1]