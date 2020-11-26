#include "Levin_power.h"
#include <fstream>
#include <chrono>

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
    Levin_power lp(10, z_bg, chi_bg, chi_cl, kernel, k_pk, z_pk, pk_l, pk_nl);

 // Define output file    
    std::fstream test;
    test.open("test.txt", std::ios::out);
    uint N = 50;
    std::vector<uint> ell(N, 0.0);
    for (uint i = 0; i < N; i++)
    {
        ell.at(i) = 2 + i;
    }
    std::vector<double> result(ell.size() * Ntomo * Ntomo, 0.0);
    auto start = high_resolution_clock::now();
    result = lp.all_C_ell(ell, true);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count()/1.0e6 << "s" << std::endl;
    for (uint i = 0; i < N; i++)
    {
        test << ell.at(i);
        for (uint j = 0; j < Ntomo * Ntomo; j++)
        {
            uint i_tomo = j / Ntomo;
            uint j_tomo = j - i_tomo * Ntomo;
            test << " " << result[i * Ntomo * Ntomo + j] << " " << lp.Limber(ell.at(i), i_tomo, j_tomo, true);
        }
        test << std::endl;
    }
    test.close();
    return 0;
}