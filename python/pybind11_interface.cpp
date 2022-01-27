#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

#include "Levin_power.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(levinpower, m)
{
     m.doc() = "Compute integrals with Levin's method.";

     py::class_<Levin_power>(m, "LevinPower")
         .def(py::init<std::vector<uint>, uint, std::vector<double>, std::vector<double>, std::vector<double>,
                       std::vector<std::vector<double>>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>,
                       bool, uint, uint,
                       double, double, double, uint, uint>(),
              "ell"_a, "number_count"_a, "z_bg"_a, "chi_bg"_a, "chi_cl"_a,
              "kernel"_a, "k_pk"_a, "z_pk"_a, "pk_l"_a, "pk_nl"_a,
              "precompute_splines"_a=false, "ell_max_non_Limber"_a=95, "ell_max_ext_Limber"_a=1000,
              "tol_rel"_a=1.0e-7, "limber_tolerance"_a=1.0e-2, "min_interval"_a=1.0e-2, "maximum_number_subintervals"_a=10,
              "n_collocation"_a=7) // Keyword arguments
         .def("all_C_ell", &Levin_power::all_C_ell,
              "ell"_a, "linear"_a,                      // Keyword arguments
              py::call_guard<py::gil_scoped_release>(), // Should (?) release GIL
              R"(Returns the spectra for all tomographic bin combinations (i<j) for a list of multipoles.
The final result is of the following shape: (l * n_total * n_total + i), where l is the index of the ell list, 
n_total is the number of tomographic bins and i = i_tomo*n_total + j_tomo.)")
         .def("init_splines", &Levin_power::init_splines,
              "z_bg"_a, "chi_bg"_a, "chi_cl"_a, "kernel"_a, "k_pk"_a, "z_pk"_a, "pk_l"_a, "pk_nl"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("compute_C_ells", &Levin_power::compute_C_ells,
              "ell_parallel"_a=false,                                   // Keyword arguments
              py::call_guard<py::gil_scoped_release>()); // Should (?) release GIL;  // Doc string
}