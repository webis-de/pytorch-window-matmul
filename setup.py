import os
import platform
import sys
import pathlib

import torch
from setuptools import setup, find_packages
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

__version__ = "0.1.0"
URL = "https://github.com/fschlatt/window_matmul"

WITH_CUDA = False
if torch.cuda.is_available():
    WITH_CUDA = CUDA_HOME is not None


def get_extension():
    setup_dir = pathlib.Path(".")
    src_dir = setup_dir / "kernel"
    src_files = []
    src_files.extend(src_dir.rglob("*.cpp"))
    src_files.extend(src_dir.rglob("*.cu"))
    # remove generated 'hip' files, in case of rebuilds
    src_files = [path for path in src_files if "hip" not in str(path)]
    # remove cuda files if cuda not available
    src_files = [path for path in src_files if path.parent.name != "cuda" or WITH_CUDA]
    src_files = [str(path) for path in src_files]

    define_macros = [("WITH_PYTHON", None)]
    undef_macros = []

    if sys.platform == "win32":
        define_macros += [("torchscatter_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    if WITH_CUDA:
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O3"]
        if torch.version.hip:
            # USE_ROCM was added to later versions of PyTorch.
            # Define here to support older PyTorch versions as well:
            define_macros += [("USE_ROCM", None)]
            undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        else:
            nvcc_flags += ["--expt-relaxed-constexpr"]
        extra_compile_args["nvcc"] = nvcc_flags

    Extension = CUDAExtension if WITH_CUDA else CppExtension
    extension = Extension(
        "kernel",
        src_files,
        include_dirs=[str(src_dir)],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension]


# work-around hipify abs paths
include_package_data = True
if torch.cuda.is_available() and torch.version.hip:
    include_package_data = False

setup(
    name="window_matmul",
    version=__version__,
    description="PyTorch extension for windowed matrix multiplication",
    author="Ferdinand Schlatt",
    author_email="ferdinand.schlatt@uni-jena.de",
    url=URL,
    # download_url=f"{URL}/archive/{__version__}.tar.gz",
    keywords=["pytorch", "matmul", "window"],
    python_requires=">=3.7",
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True, use_ninja=False
        )
    },
    packages=find_packages(),
    include_package_data=include_package_data,
)
