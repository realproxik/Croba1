#include <torch/extension.h>

torch::Tensor square_relu_cuda(torch::Tensor input);

torch::Tensor square_relu(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  return square_relu_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("square_relu", &square_relu, "Croba squared ReLU (CUDA)");
}

