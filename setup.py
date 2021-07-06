from setuptools import setup, find_packages

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from distutils.command.build_ext import build_ext
import sys
import sysconfig
import os


def locate_cuda():
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = os.path.join(home, "bin", "nvcc")
    else:
        for directory in os.environ["PATH"].split(os.pathsep):
            nvcc = os.path.join(directory, "nvcc")
            if os.path.exists(nvcc):
                # further check whether the header files and library are there
                home = os.path.dirname(os.path.dirname(nvcc))
                if os.path.exists(os.path.join(home, "include")) and os.path.exists(
                    os.path.join(home, "lib64")
                ):
                    cudaconfig = {
                        "home": home,
                        "nvcc": nvcc,
                        "include": os.path.join(home, "include"),
                        "lib64": os.path.join(home, "lib64"),
                    }
                    return cudaconfig
        return None


def customize_compiler_for_nvcc(self):
    """
    inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class cuda_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def find_files_with_ext(path, ext):
    files_list = list()
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                current_ext = name.split(".")[-1]
            except IndexError:
                continue

            if current_ext == ext:
                files_list.append(join(root, name))
    return files_list


suffix = sysconfig.get_config_var("EXT_SUFFIX")
if suffix is None:
    suffix = ".so"

extra_link_args = []
if sys.platform == "darwin":
    from distutils import sysconfig

    vars = sysconfig.get_config_vars()
    vars["LDSHARED"] = vars["LDSHARED"].replace("-bundle", "-shared")
    extra_link_args = ["-Wl,-install_name,@rpath/libabie" + suffix]

sources = [
    "src/integrator_gauss_radau15.c",
    "src/integrator_wisdom_holman.c",
    "src/integrator_runge_kutta.c",
    "src/common.c",
    "src/additional_forces.c",
]

# Locate CUDA paths
CUDA = locate_cuda()

gcc_compile_flags = [
    "-fstrict-aliasing",
    "-O3",
    "-std=c99",
    "-fPIC",
    "-shared",
    "-fcommon",
    "-fcommon",
]
if CUDA is not None:
    # CUDA/GPU detected on the system
    sources.append("src/gpuforce.cu")
    sources.append("src/bodysystemcuda.cu")
    extra_link_args = ["-fPIC", "-DGPU", "-O3", "-g", "-lcudart", "-lstdc++"]
    gcc_compile_flags.append("-DGPU")
    print("CUDA SDK found! ABIE will be built with GPU support.")
else:
    print("CUDA SDK  not found! ABIE will be built without GPU support.")

module_abie = Extension(
    name="libabie",
    sources=sources,
    extra_link_args=extra_link_args,
    extra_compile_args={
        "gcc": gcc_compile_flags,
        "nvcc": [
            "-O3",
            "-Xcompiler",
            "-DGPU",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-c",
            "--compiler-options",
            "-fPIC",
        ],
    },
)

setup(
    name="abie",
    version="0.7.5",
    description="Alice-Bob Integrator Environment (ABIE), a GPU-accelerated integrator framework for astrophysical N-body simulations",
    url="https://github.com/MovingPlanetsAround/ABIE",
    author="Maxwell X. Cai, Javier Roa, Adrian S. Hamers, Nathan W. C. Leigh",
    author_email="maxwellemail@gmail.com",
    license="GPL-3.0",
    packages=find_packages(),
    zip_safe=False,
    install_requires=["toml", "numpy", "h5py"],
    entry_points={"console_scripts": ["abie = ABIE.abie:main"]},
    cmdclass={
        "build_ext": cuda_build_ext,
    },
    ext_modules=[module_abie],
)
