from setuptools import setup, find_packages
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
# from distutils.core import setup, Extension
import sys
import sysconfig
import os
# name is the package name in pip, should be lower case and not conflict with existing packages
# packages are code source


suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

extra_link_args=['-fcommon']
if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-shared')
    extra_link_args=['-Wl,-install_name,@rpath/libabie'+suffix]

module_abie = Extension(name = 'libabie',
                        sources = ['src/integrator_gauss_radau15.c',
                            'src/integrator_wisdom_holman.c',
                            'src/integrator_runge_kutta.c',
                            'src/common.c',
                            'src/additional_forces.c'],
                        include_dirs = ['src'],
                        extra_compile_args=['-fstrict-aliasing', '-O3','-std=c99','-fPIC', '-shared', '-fcommon'],
                        extra_link_args=extra_link_args,
                        )

setup(name='abie',
      version='0.3.2',
      description='Alice-Bob Integrator Environment (ABIE), a GPU-accelerated integrator framework for astrophysical N-body simulations',
      url='https://github.com/MovingPlanetsAround/ABIE',
      author='Maxwell X. Cai, Javier Roa, Adrian S. Hamers, Nathan W. C. Leigh',
      author_email='maxwellemail@gmail.com',
      license='BSD 2-Clause',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['toml', 'numpy', 'h5py'],
      entry_points={'console_scripts': ['abie = ABIE.abie:main'] },
      ext_modules = [module_abie],
      )
