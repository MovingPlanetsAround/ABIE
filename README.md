[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/abie)](https://pepy.tech/project/abie)
[![CodeQL](https://github.com/MovingPlanetsAround/ABIE/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MovingPlanetsAround/ABIE/actions/workflows/codeql-analysis.yml)
# Moving Planets Around (MPA) Project

*Moving Planets Around* is an education book project that teaches students to build a state-of-the-art *N*-Body code for planetary system dynamics from the stretch. The code built throughout the storyline of the book is hosted here. The book has been published by the MIT Press in September 2020. See: https://mitpress.mit.edu/books/moving-planets-around 

## The Alice-Bob Integrator Environment (ABIE)
------
In the book *Moving Planets Around*, Alice and Bob are two students interested in building an *N*-body code. With the help from Prof. Starmover, they eventually achieved their ambition. The resulting source code `ABIE` is named after them.

`ABIE` is a general-purpose direct *N*-body library for astrophysical simulations. It is especially well-suited for simulating planetary systems with or without a large number of test particles (e.g., comets, asteroids). `ABIE` implements all its integrators in both Python (for educational purpose) and in C (for performance). The currently supported integrators are:

- Forward Euler integrator 
- Leapfrog
- Adams-Bashforth
- Runge-Kutta
- Gauss-Radau15 *(default)*
- Wisdom-Holman

In the scenarios where the number of particles is large, ABIE makes use of the GPUs (if available) to accelerate the calculation of gravity.

## Installation
`ABIE` is a Python library with C backend. Like most Python packages, it can simply be installed with 

    pip install abie


## Getting Started

### Setup a simple planetary system
A simple `ABIE` simulation can be created like this:
```python
import abie
import numpy as np

# create an ABIE instance (Units: MSun, au, yr)
sim = abie.ABIE(CONST_G=4*np.pi**2, name="example_simulation")

# select the integrator
sim.integrator = "WisdomHolman" 

# add particles
sim.add(mass=1.0, pos=[0., 0., 0.,], vel=[0., 0., 0.,], name="Sun")
sim.add(mass=3.e-6, a=1.0, name='Earth', primary='Sun')
# Use `sim.add()` to add as many particles as needed

# print an overview of particles
print(sim.particles)

# initialize the integrator
sim.initialize()

# perform the integration for 1000 years
sim.integrate(1000)
sim.stop()
```

In the example above, a simple planetary system is setup, such that a solar-type star named "Sun" is orbited by an Earth-mass planet at a=1 au. A particle can either be added using Cartesian coordinates or orbital elements. The Sun is sitting at the center of the reference frame with zero initial velocity. When the simulation is done, you will find a HDF5 file named `example_simulation.h5` containing all the simulation data. 


### Setup a hierarchical system
It is also very straightforward to set up hierarchical systems. For examiple, consider a moon orbiting a planet, and that particular planet orbits the center-of-mass of a circumbinary stellar system:

```python
sim.add(mass=1.0, pos=[0., 0., 0.,], vel=[0., 0., 0.,], name="star1") # add the first star
sim.add(mass=1.0, a=1, name="star2", primary='star1') # add the second star
sim.particles["star1"].primary = sim.particles["star2"] # make the second star also orbit the first one

# make a planet orbiting the center-of-mass of star1 and star2
sim.add(mass=1.e-5, a=10, e=0.2, i=0.1, name='planet', primary=['star1', 'star2']) 

# make a moon orbiting the planet
sim.add(mass=1.e-9, a=0.01, e=0.0, i=0.0, name='moon', primary='planet') 
```

### Obtain the simulation data
All simulation data can be found in the HDF5 output file. Alternative, it is also possible to use the `recorder` facility to record the simulation data. For example, if we are running a Jupyter notebook and would like to plot the orbital elements, we could do

```python    
# tell the recorder that semi-major axes, eccentricities, inclinations, time, and energy error should be recorded
sim.record_simulation(quantities=["a", "ecc", "inc", "time", "energy"]).start()

# perform the integration for some time
sim.integrate(200)
```

After this, the simulation data can be accessed with `sim.data`, which is a dictionary. For example, if we would like to plot the orbital eccentrcities evolution, we could simply do
```python
import matplotlib.pyplot as plt

plt.plot(sim.data['time'], sim.data['ecc'])
plt.show()
```
Likewise, generate a 2D scatter plot of x-y would be as easy as 

```python
sim.record_simulation(quantities=["x", "y", "z", "time"]).start()

# perform the integration for some time
sim.integrate(200)

plt.scatter(sim.data['x'], sim.data['y'])
```




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
semi = h5f['/Step#0/semi'][()]
ecc = h5f['/Step#0/ecc'][()]
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
semi = h5f['/semi'][()] # gets the semi-axis array for the entire simulation
ecc = h5f['/ecc'][()]    # gets the eccentricity array for the entire simulation
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

By default, `ABIE` uses double precision. For some special cases (for example, integrating a Kozai-Lidov system where the eccentricity can be very high), the precision of the integrator can be adjusted by simply changing the following lines in `src/Makefile` from

```Makefile
LONGDOUBLE = 0
```
    
to 

```Makefile
LONGDOUBLE = 1
```

And run `make clean; make` again. This  will causes the integrator to use the [`long double`](https://en.wikipedia.org/wiki/Long_double) data type. When using the Gauss-Radau15 integrator, the energy conservation can be better than 10^{-16} in this case (shown as `dE/E = 0` in the terminal), even after evolving the system through multiple Kozai cycles. This, however, will takes about 2x time to finish evolving the same system.


### Accelerate ABIE using CUDA/GPU

When `ABIE` is being installed, the `setup.py` script will automatically determine whether your machine has CUDA GPUs or not. If GPUs are presented in the machine, it will automatically turn on the relevant compilation flags. This could also be done manually if you wish to use `ABIE` as a C library instead. For large `N` systems (N>512), using the GPU could result in substential speed up. To enable the GPU support, modify the `src/Makefile` from

```Makefile
GPU = 0
```
    
to 

```Makefile
GPU = 1
```

And then recompile the code. Note that it is not recommended to use GPU for small-N systems. Using GPU on small-N system is actually slower than using the CPU.

  


