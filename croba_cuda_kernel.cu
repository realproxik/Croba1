#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void square_relu_kernel(const scalar_t* input, scalar_t* output, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const scalar_t zero = scalar_t(0);
    const scalar_t value = input[i] > zero ? input[i] : zero;
    output[i] = value * value;
  }
}

torch::Tensor square_relu_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int threads = 256;
  const int blocks = (input.numel() + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "square_relu_cuda", [&] {
    square_relu_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
