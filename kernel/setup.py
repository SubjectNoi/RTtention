from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {
    "cxx": [
        "-std=c++17",
        "-O2"
    ],
    "nvcc": [
        "-O2",
        "-std=c++17",
        "-arch=compute_80",
        "-code=sm_80",
        "-lineinfo",
    ],
}

setup(
    name="RTtention",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="RTtention",
            sources=[
                "pybind.cpp",
                "rt_gemv/rt_gemv.cu",
            ],
            include_dirs=[
                "/home/zhliu/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/include",
                "/home/zhliu/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK",
                "/home/zhliu/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/support",
                "/home/zhliu/workspace/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/build",
                "/home/zhliu/workspace/RTtention/kernel/rt_gemv/include",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)

