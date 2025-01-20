#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "rt_gemv/rt_gemv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rt_gemv", &rt_gemv, "");
}
