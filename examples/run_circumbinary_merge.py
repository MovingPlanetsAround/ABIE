"""
Run ABIE programmatically as a library.
"""
import abie
import numpy as np

# create an ABIE instance
sim = abie.ABIE(name="circumbiary")

# Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
# sim.integrator = 'WisdomHolman'
sim.integrator = "GaussRadau15"
# sim.integrator = 'RungeKutta'

# The termination time (optional; can be overridden by the integrate() function)

# The underlying implementation of the integrator ('ctypes' or 'numpy')
sim.acceleration_method = "ctypes"
# sim.acceleration_method = 'numpy'

# Add the objects
# Create a circumbinary system. The planet orbits around the center-of-mass of the two star, and the moon orbits around the planet. Due to the Kozai-Lidov effect, the moon will soon collide with the planet and merged into it.

sim.add(mass=0.6897, x=0, y=0, z=0, vx=0, vy=0, vz=0, name="star1")
sim.add(
    mass=0.20255,
    a=0.22431,
    e=0.15944,
    i=1.57673219158,
    omega=4.59831426047,
    Omega=0.0,
    f=1.0e-5,
    name="star2",
    primary="star1",
    radius=1.0e-3,
)
sim.particles["star1"].primary = sim.particles["star2"]
sim.add(
    mass=0.000317894686564,
    a=0.7048,
    e=0.0069,
    i=1.57135832281,
    omega=5.55014702134,
    Omega=5.23598775598e-05,
    f=1.0e-5,
    name="planet",
    primary=["star1", "star2"],
    radius=1.0e-4,
)
sim.add(
    mass=5.02785431289e-08,
    a=0.00884615384615,
    e=0.01,
    i=-0.0510158147621,
    omega=0.0,
    Omega=3.14164501347,
    f=4.86470951828,
    name="moon",
    primary="planet",
    radius=1.0e-5,
)

print(sim.particles)


# The output frequency
sim.store_dt = 1

# The integration timestep (does not apply to Gauss-Radau15)

# The size of the buffer. The buffer is full when `buffer_len` outputs are
# generated. In this case, the collective output will be flushed to the HDF5
# file, generating a `Step#n` HDF5 group
sim.buffer_len = 10000

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

# perform the integration
sim.integrate(300)

sim.stop()
