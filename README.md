[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/abie)](https://pepy.tech/project/abie)
# Moving Planets Around (MPA) Project

*Moving Planets Around* is an education book project that teaches students to build a state-of-the-art N-Body code for planetary system dynamics from the stretch. The code built throughout the storyline of the book is hosted here. The book has been published by the MIT Press in September 2020. See: https://mitpress.mit.edu/books/moving-planets-around 

## The Alice-Bob Integrator Environment (ABIE)
------
`ABIE` is a general-purpose direct N-body library for astrophysical simulations. It is especially well-suited for simulating planetary systems with or without a large number of test particles (e.g., comets, asteroids). ABIE implements all its integrators in both Python (for educational purpose) and in C (for performance). The currently supported integrators are:

- Forward Euler integrator 
- Leapfrog
- Adams-Bashforth
- Runge-Kutta
- Gauss-Radau15 *(default)*
- Wisdom-Holman

In the scenarios where the number of particles is large, ABIE makes use of the GPUs (if available) to accelerate the calculation of gravity.

## Installation
ABIE can be installed through 

    python setup.py install
Note that `setup.py` is in the parent directory of `ABIE`'s main source code directory. Example scripts can be found in the `/examples` directory. Alternatively, you could install ABIE with `pip install abie`.

## ABIE output format

`ABIE` uses the HDF5 format to store its integration output data. The internal layout of the HDF5 file looks like this:

    /Step#0
        ecc: eccentricities of each object (1 - N) as a function of time (0, t-1): [t * N] array
        inc: inclinations of each object (1 - N) as a function of time (0, t-1): [t * N] array
        hash: unique integer identifier for each particle: [t * N] array
        mass: masses of each planet: [t * N] array
        ptype: particle type of each object. 0 = massive particles; 1 = test particles; 2 = low-mass particles. [t * N] array
        radius: radius of each object. [t * N] array
        semi: semi-major axis of each object. [t * N] array
        time: the time vector. [t * 1] vector
        vx: [t * N]
        vy: [t * N]
        vz: [t * N]
        x: [t * N]
        y: [t * N]
        z: [t * N]
    /Step#1
        ...
    /Step#2
        ...
    ...
    /Step#n
    
For efficient output, ABIE maintains a buffer to temporarily store its integration data. The default buffer length is set to 1024, which means that it will accumulate 1024 snapshot output until it flushes the data into the HDF5 file and creates a `Step#n` group. The resulting HDF5 datasets can be loaded easily using the `h5py` package:

```python
import h5py
h5f = h5py.File('data.hdf5', 'r')
semi = h5f['/Step#0/semi'].value
ecc = h5f['/Step#0/ecc'].value
...
h5f.close()
```    

Sometimes, it is more convenient to get rid of the `Step#n` data structure in the HDF5 file (i.e. combine `Step#0`, `Step#1`, ..., `Step#n` into flatten arrays). The `ABIE` package contains a tool to seralize the snapshot. For example, suppose that `ABIE` generates a data file `data.hdf5` contains the `Step#n` structure, the following command

```
python snapshot_serialization.py -f data.hdf5
```
will generate a flattened file called `data` (still in hdf5 format). In this case, the data can be accessed in this way:
```python
import h5py
h5f = h5py.File('data.hdf5', 'r')
semi = h5f['/semi'].value  # gets the semi-axis array for the entire simulation
ecc = h5f['/ecc'].value    # gets the eccentricity array for the entire simulation
...
h5f.close()
```    

## Computational Acceleration

By default, ABIE will execute the C implementation of the Gauss-Radau15 integrator. This integrator is well-optimized and preserves energy to ~ 10^{-15}. It is very straightforward to change the integrator from the Python interface:

```python
from abie import ABIE
sim = ABIE()
sim.CONST_G = 1.0
sim.integrator = 'GaussRadau15' # or 'WisdomHolman', 'LeapFrog', etc.
```
    
## Improve the precision of ABIE

By default, `ABIE` uses double precision. For some special cases (for example, integrating a Kozai-Lidov system where the eccentricity can be very high), the precision of the integrator can be adjusted by simply changing the following lines in `Makefile` from

```Makefile
LONGDOUBLE = 0
```
    
to 

```Makefile
LONGDOUBLE = 1
```

And run `make clean; make` again. This  will causes the integrator to use the [`long double`](https://en.wikipedia.org/wiki/Long_double) data type. When using the Gauss-Radau15 integrator, the energy conservation can be better than 10^{-16} in this case (shown as `dE/E = 0` in the terminal), even after evolving the system through multiple Kozai cycles. This, however, will takes about 2x time to finish evolving the same system.


### Accelerate ABIE using CUDA/GPU

`ABIE` supports GPU acceleration. For large `N` systems (N>512), using the GPU could result in substential speed up. To enable the GPU support, modify the `Makefile` from

```Makefile
GPU = 0
```
    
to 

```Makefile
GPU = 1
```

And then recompile the code. Note that it is not recommended to use GPU for small-N systems. Using GPU on small-N system is actually slower than using the CPU.

  


