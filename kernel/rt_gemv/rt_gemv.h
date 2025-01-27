#pragma once
#include <torch/extension.h>

torch::Tensor rt_gemv(
	torch::Tensor in,
	torch::Tensor quantized_w,
	torch::Tensor codebook, 
	torch::Tensor centers,
    torch::Tensor radius, 
	torch::Tensor origins
);
