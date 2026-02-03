#include "tokenizer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

PYBIND11_MODULE(clip_tokenizer_cpp_py, m) {
    pybind11::class_<SimpleTokenizer>(m, "CLIPTokenizer")
        .def(pybind11::init<>())
        .def("encode", &SimpleTokenizer::encode, pybind11::arg("text"))
        .def("decode", &SimpleTokenizer::decode, pybind11::arg("tokens"));
    
} 