"""
Run ABIE programmatically as a library.
"""
import abie
import numpy as np

# create an ABIE instance
sim = abie.ABIE(CONST_G=4 * np.pi ** 2, name="trappist1")

# Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
sim.integrator = "WisdomHolman"
# sim.integrator = 'GaussRadau15'
# sim.integrator = 'RungeKutta'

# Use the CONST_G parameter to set units
sim.CONST_G = 4 * np.pi ** 2  # Units: AU, MSun, yr


# The underlying implementation of the integrator ('ctypes' or 'numpy')
sim.acceleration_method = "ctypes"
# sim.acceleration_method = 'numpy'

# Add the objects: create a TRAPPIST-1 system. Data from Wikipedia

me_ms = 3.0e-6  # mass ratio between the Earth and the Sun

sim.particles.add(mass=0.0898, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), name="star")
sim.particles.add(
    mass=1.374 * me_ms,
    a=0.01154,
    e=0.00622,
    i=np.radians(89.56),
    name="b",
    primary="star",
    f=2 * np.pi * np.random.rand(),
)
sim.particles.add(
    mass=1.308 * me_ms,
    a=0.01580,
    e=0.01580,
    i=np.radians(89.70),
    name="c",
    primary="star",
    f=2 * np.pi * np.random.rand(),
)
sim.particles.add(
    mass=0.388 * me_ms,
    a=0.02227,
    e=0.00837,
    i=np.radians(89.89),
    name="d",
    primary="star",
    f=2 * np.pi * np.random.rand(),
)
sim.particles.add(
    mass=0.692 * me_ms,
    a=0.02925,
    e=0.00510,
    i=np.radians(89.736),
    name="e",
    primary="star",
    f=2 * np.pi * np.random.rand(),
)
sim.particles.add(
    mass=1.039 * me_ms,
    a=0.03849,
    e=0.01007,
    i=np.radians(89.719),
    name="f",
    primary="star",
    f=2 * np.pi * np.random.rand(),
)
sim.particles.add(
    mass=1.321 * me_ms,
    a=0.04683,
    e=0.00208,
    i=np.radians(89.721),
    name="g",
    primary="star",
    f=2 * np.pi * np.random.rand(),
)
sim.particles.add(
    mass=0.326 * me_ms,
    a=0.06189,
    e=0.00567,
    i=np.radians(89.796),
    name="h",
    primary="star",
    f=2 * np.pi * np.random.rand(),
)

print(sim.particles)


# The output frequency
sim.store_dt = 0.1

# The integration timestep (does not apply to Gauss-Radau15)
sim.h = 0.001

# The size of the buffer. The buffer is full when `buffer_len` outputs are
# generated. In this case, the collective output will be flushed to the HDF5
# file, generating a `Step#n` HDF5 group

# set primary body for the orbital element calculation
# `#COM#`: use the center-of-mass as the primary body
# `#M_MAX#`: use the most massive object as the primary object
# `#M_MIN#`: use the least massive object as the primary object
# One could also specify the name of the object (e.g, 'Sun', 'planet')
# or the ID of the object (e.g., 0, 1), or a list of IDs / names defining
# a subset of particles.
# sim.particles.primary = [0, 1]

# initialize the integrator
sim.initialize()

sim.record_simulation(quantities=["a", "ecc", "inc", "time", "energy"]).start()

# perform the integration
sim.integrate(200)


print(sim.data)
sim.stop()
