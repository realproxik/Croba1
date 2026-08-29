from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="croba_cuda",
    ext_modules=[CUDAExtension("croba_cuda", ["croba_cuda.cpp", "croba_cuda_kernel.cu"])],
    cmdclass={"build_ext": BuildExtension},
)

