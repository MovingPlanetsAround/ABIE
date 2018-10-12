"""
Run ABIE programmatically as a library.
"""
try:
    from ABIE.abie import ABIE
except ImportError:
    from abie import ABIE
import numpy as np

# create an ABIE instance
sim = ABIE()

# Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
# sim.integrator = 'WisdomHolman'
sim.integrator = 'GaussRadau15'
# sim.integrator = 'RungeKutta'

# Use the CONST_G parameter to set units
sim.CONST_G = 4 * np.pi ** 2
# sim.CONST_C = 63198.0
# sim.CONST_G = 1

# The termination time (optional; can be overridden by the integrate() function)
sim.t_end = 1000

# The underlying implementation of the integrator ('ctypes' or 'numpy')
sim.acceleration_method = 'ctypes'
# sim.acceleration_method = 'numpy'

# Add the objects
# sim.add(mass=1, x=0, y=0, z=0, vx=0, vy=0, vz=0, name='Sun')
# sim.add(mass=1.e-3, a=1, e=0.0, name='planet', primary='Sun')
# sim.add(mass=1.e-8, a=0.01, e=0.2, name='moon', primary='planet')
# sim.particles['Sun'].primary = 'planet'

# sim.add(mass=1, x=0, y=0, z=0, vx=0, vy=0, vz=0, name='star1')
# sim.add(mass=0.0001, a=1, e=0.9500000001, name='star2', primary='star1', radius=0.0501)
# sim.particles['star1'].primary = 'star2'
# sim.add(mass=1.e-3, a=10, e=0, primary=['star1', 'star2'])
# print sim.particles
# sys.exit(0)
# sim.add(mass=1, x=1, y=0, z=0, vx=0, vy=2.44948974, vz=0, name='planet')
# sim.add(mass=0.6897, x=-0.00468615, y=0.00025252, z=-0.04268986, vx=3.32104666e+00, vy=2.25621462e-03, vz=-3.78372988e-01)
# sim.add(mass=0.20255, x=0.0167729, y=-0.00085939, z=0.1446277, vx=-1.13009713e+01, vy=-7.68687748e-03, vz=1.29669470e+00)
# sim.add(mass=0.000317894686564, x=-5.19974700e-01, y=-2.90347439e-04, z=4.68177011e-01, vx=-4.76228444e+00, vy=2.47017879e-03, vz=-5.28915445e+00)
# sim.add(mass=5.02785431289e-08, x=-5.15024957e-01, y=-2.39607087e-04, z=4.68175573e-01, vx=-4.77853383, vy=1.60208426, vz=-5.33471724)

# sim.add(mass=0.6897, x=-0.00468615, y=0.00025252, z=-0.04268986, vx=3.32104666e+00, vy=2.25621462e-03, vz=-3.78372988e-01, name='star1')
# sim.add(mass=0.20255, x=0.01677295, y=-0.00085939, z=0.1446277, vx=-1.13009713e+01, vy=-7.68687748e-03, vz=1.29669470e+00, name='star2')
# sim.add(mass=0.000317894686564, x=-5.19973162e-01, y=-2.90081192e-04, z=4.68177003e-01, vx=-4.76236811e+00, vy=2.95761444e-03, vz=-5.28916834e+00, name='planet', primary=['star1', 'star2'])
# sim.add(mass=5.02785431289e-08, x=-0.52474875, y=-0.00192299, z=0.46822351, vx=-4.24953552, vy=-1.47981103, vz=-5.24693299, name='moon', primary='planet')

# sim.close_encounter_distance = 0
# sim.max_close_encounter_events = 100
# sim.max_collision_events = 1

sim.add(mass=0.6897, x=0, y=0, z=0, vx=0, vy=0, vz=0, name='star1')
sim.add(mass=0.20255, a=0.22431, e=0.15944, i=1.57673219158, omega=4.59831426047, Omega=0.0, f=1.e-5, name='star2', primary='star1', radius=1.e-3)
sim.particles['star1'].primary = sim.particles['star2']
sim.add(mass=0.000317894686564, a=0.7048, e=0.0069, i=1.57135832281, omega=5.55014702134, Omega=5.23598775598e-05, f=1.e-5, name='planet', primary=['star1', 'star2'], radius=1.e-4)
# sim.add(mass=5.02785431289e-08, a=0.00884615384615, e=0.01, i=0.50158147621, omega=0.0, Omega=3.14164501347, f=4.86470951828, name='moon', primary='planet')
sim.add(mass=5.02785431289e-08, a=0.00884615384615, e=0.01, i=-0.0510158147621, omega=0.0, Omega=3.14164501347, f=4.86470951828, name='moon', primary='planet', radius=1.e-5)
# sim.add(mass=5.02785431289e-08, a=0.005, e=0.01, i=-np.pi/2, omega=0.0, Omega=3.14164501347, f=4.86470951828, name='moon', primary='planet')
print sim.particles
# sys.exit(0)
# sim.add(mass=)
# The output file name. If not specified, the default is 'data.hdf5'
sim.output_file = 'abc.h5'
sim.collision_output_file = 'abc.collisions.txt'
sim.close_encounter_output_file = 'abc.ce.txt'

# The output frequency
sim.store_dt = 0.1

# The integration timestep (does not apply to Gauss-Radau15)
sim.h = 0.001

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
sim.integrate(30)  # the argument `2000` is optional if `sim.t_end` is specified in the context

sim.stop()
