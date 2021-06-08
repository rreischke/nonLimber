#include "Levin_power.h"
#include <fstream>
#include <chrono>
#include <boost/math/special_functions/bessel.hpp>

using namespace std::chrono;

int main()
{
    /**
     * Sets up the input for the integration class.
     */
    std::vector<double> z_bg;
    std::vector<double> chi_bg;
    std::vector<double> chi_cl;
    std::vector<std::vector<double>> kernel;
    std::vector<double> k_pk;
    std::vector<double> z_pk;
    std::vector<double> pk_l;
    std::vector<double> pk_nl;

    uint Nz = 50;
    uint Nk = 200;
    uint Nbg = 1024;
    uint Nkernel = 2000;
    uint Ntomo = 15;
    std::fstream in_bg, in_kernel, in_pk, in_kernel_gc;
    in_bg.open("bg.txt", std::ios::in);
    for (uint i = 0; i < Nbg; i++)
    {
        double chi, z;
        in_bg >> z >> chi;
        z_bg.push_back(z);
        chi_bg.push_back(chi);
    }
    in_bg.close();
    in_kernel.open("kernels_sh.txt", std::ios::in);
    in_kernel_gc.open("kernels_gc.txt", std::ios::in);
    uint k = 0;
    while (in_kernel)
    {
        double chi;
        in_kernel_gc >> chi;
        in_kernel >> chi;
        chi_cl.push_back(chi);
        kernel.push_back(std::vector<double>());
        for (uint j = 0; j < Ntomo; j++)
        {
            double kernel_dummy;
            if (j < 10)
            {
                in_kernel_gc >> kernel_dummy;
            }
            else
            {
                in_kernel >> kernel_dummy;
            }
            kernel.at(k).push_back(kernel_dummy);
        }
        k++;
    }
    in_kernel.close();
    in_kernel_gc.close();
    in_pk.open("pk.txt", std::ios::in);
    for (uint i = 0; i < Nk * Nz; i++)
    {
        double k_aux, z_aux, pk_l_aux, pk_nl_aux;
        in_pk >> z_aux >> k_aux >> pk_l_aux >> pk_nl_aux;
        pk_l.push_back(pk_l_aux);
        pk_nl.push_back(pk_nl_aux);
        if (i < Nk)
        {
            k_pk.push_back(k_aux);
        }
        if (i % Nk == 0)
        {
            z_pk.push_back(z_aux);
        }
    }
    in_pk.close();
    // Input definitions done

    // Constructing the input of the class

    Levin_power lp(false, 10, z_bg, chi_bg, chi_cl, kernel, k_pk, z_pk, pk_l, pk_nl);
    
    // Define output file
    std::fstream test;
    test.open("./test.txt", std::ios::out);
    uint N = 10000;
    double kmin = 1e-3;
    double kmax = 1.0;
    //lp.levin_integrate_bessel(10, 2, 5, false);
    //lp.levin_integrate_bessel(1e-1, 30, 9, false);
    auto start = high_resolution_clock::now();
    lp.set_auxillary_splines(3, false);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1.0e6 << "s" << std::endl;

    //std::cout << lp.C_ell_full(6, 6) << std::endl;
    for (uint i = 0; i < N; i++)
    {
        uint ell = 2 + i;
        //lp.set_auxillary_splines(ell, false);
        double k = exp(log(kmin) + (log(kmax) - log(kmin)) / (N - 1.0) * i);
        //test << ell << " " << lp.Limber(ell,0,3,false) << " " << lp.extended_Limber(ell,0,3) << std::endl;
        test << k;
        for (uint j = 0; j < 15; j++)
        {
            test << " " << lp.auxillary_weight(j, k);
        }
        test << std::endl;
        //test << ell << " " << lp.C_ell_full(6, 6) << std::endl;
    }
    /*
    uint M = 5;
    for (uint i = 0; i < M; i++)
    {
        double final = 0.0;
        lp.set_auxillary_splines(i+2,false);
#pragma omp parallel for
        for (uint j = 0; j < N; j++)
        {
            double k = exp(log(kmin) + (log(kmax) - log(kmin)) / (N - 1.0) * j);
            double dk = (log(kmax) - log(kmin)) / (N - 1.0) * k;
            double bessel = lp.levin_integrate_bessel(k, i + 2, 9, false);
            final += bessel * bessel * dk;
        }
        test << i + 2 << " " << final << " " << lp.C_ell_full(9,9) << std::endl;
    }
*/
    /*uint N = 1998;
    double ellmin = 2;
    double ellmax = 2000;
    double elllin = 10;
    std::vector<uint> ell(N, 0.0);
    for (uint i = 0; i < N; i++)
    {
        ell.at(i) = i + ellmin;
    }
    std::vector<double> result(ell.size() * Ntomo * Ntomo, 0.0);
    auto start = high_resolution_clock::now();
    result = lp.all_C_ell(ell, false);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1.0e6 << "s" << std::endl;
    for (uint i = 0; i < N; i++)
    {
        std::cout << i << std::endl;
        test << ell.at(i);
        for (uint j = 0; j < Ntomo * Ntomo; j++)
        {
            uint i_tomo = j / Ntomo;
            uint j_tomo = j - i_tomo * Ntomo;
            test << " " << result[i * Ntomo * Ntomo + j];// << " " << lp.Limber(ell.at(i), i_tomo, j_tomo, false);
        }
        test << std::endl;
    }*/
    /*  std::vector<uint> use_limber(Ntomo);
    for (uint i = 0; i < Ntomo; i++)
    {
        use_limber.at(i) = 0;
    }
    //lp.set_auxillary_splines(use_limber, 50, false);
    N = 500;
    double kmin = 1.0e-4;
    double kmax = 1.0e0;
    for (uint j = 0; j < N; j++)
    {
        double k = exp(log(kmin) + (log(kmax) - log(kmin)) / (N - 1.0) * j);
        test << k << " ";
        for (uint i = 9; i < 10; i++)
        {
            //  test << " " << lp.auxillary_weight(i, k);
            test << lp.levin_integrate_bessel(k, 10, i, true)
                 << " " << lp.levin_integrate_bessel(k, 20, i, true)
                 << " " << lp.levin_integrate_bessel(k, 40, i, true)
                 << " " << lp.levin_integrate_bessel(k, 80, i, true)
                 << " " << lp.levin_integrate_bessel(k, 160, i, true)
                 << " " << lp.levin_integrate_bessel(k, 320, i, true)
                 << " " << lp.levin_integrate_bessel(k, 760, i, true)
                 << " " << lp.levin_integrate_bessel(k, 1500, i, true);
        }
        test << std::endl;
    }*/
    test.close();
    return 0;
}