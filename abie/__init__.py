"""
Alice-Bob Integrator Environment (ABIE), created by Alice and Bob in the Moving Planets Around (MPA) project.

Features:
    1. The integrator takes input either from the commandline, or from the config file
    2. The integrator periodically stores data to HDF5
    3. The integrator supports restarting simulations


The MPA team, 2017-2018
"""
import argparse
import toml
from abie.integrator import Integrator
import numpy as np
import sys
from abie.data_io import DataIO


class ABIE(object):

    def __init__(self):
        # =================== CONSTANTS ==================
        # by default, using the square of the Gaussian gravitational constant
        self.__CONST_G = 0.000295912208232213  # units: (AU^3/day^2)
        self.__CONST_C = 0.0  # speed of light; PN terms will be calculated if CONST_C > 0

        # # =================== VARIABLES ==================
        self.__t = 0.0
        self.__t_start = 0.0
        self.__t_end = 0.0
        self.__h = 0.0
        self.__store_dt = 100  # output is triggered per 100 time units
        self.__buffer_len = 1024  # size of dataset in a hdf5 group
        self.__max_close_encounter_events = 1
        self.__max_collision_events = 1
        self.__close_encounter_distance = 0.0
        # self.acceleration_method = 'numpy'

        # load integrator modules
        self.__integrators = None  # a collection of integrators loaded from modules
        self.__integrator = None  # the actual, active integrator instance
        self.output_file = 'data.hdf5'
        self.__close_encounter_output_file = 'close_encounters.txt'
        self.__collision_output_file = 'collisions.txt'

    @property
    def max_close_encounter_events(self):
        if self.__integrator is not None:
            self.__max_close_encounter_events = self.__integrator.max_close_encounter_events
            return self.__max_close_encounter_events
        else:
            return self.__max_close_encounter_events

    @max_close_encounter_events.setter
    def max_close_encounter_events(self, value):
        self.__max_close_encounter_events = value
        if self.__integrator is not None:
            self.__integrator.max_close_encounter_events = value

    @property
    def max_collision_events(self):
        if self.__integrator is not None:
            self.__max_collision_events = self.__integrator.collision_events
            return self.__max_collision_events
        else:
            return self.__max_collision_events

    @max_collision_events.setter
    def max_collision_events(self, value):
        self.__max_collision_events = value
        if self.__integrator is not None:
            self.__integrator.max_collision_events = value

    @property
    def close_encounter_distance(self):
        if self.__integrator is not None:
            self.__close_encounter_distance = self.__integrator.max_close_encounter_events
            return self.__close_encounter_distance
        else:
            return self.__close_encounter_distance

    @close_encounter_distance.setter
    def close_encounter_distance(self, value):
        self.__close_encounter_distance = value
        if self.__integrator is not None:
            self.__integrator.close_encounter_distance = value

    @property
    def close_encounter_output_file(self):
        if self.__integrator is not None:
            self.__close_encounter_output_file = self.__integrator.close_encounter_output_file
            return self.__close_encounter_output_file
        else:
            return self.__close_encounter_output_file

    @close_encounter_output_file.setter
    def close_encounter_output_file(self, value):
        self.__close_encounter_output_file = value
        if self.__integrator is not None:
            self.__integrator.close_encounter_output_file = value

    @property
    def collision_output_file(self):
        if self.__integrator is not None:
            self.__collision_output_file = self.__integrator.collision_output_file
            return self.__collision_output_file
        else:
            return self.__collision_output_file

    @collision_output_file.setter
    def collision_output_file(self, value):
        self.__collision_output_file = value
        if self.__integrator is not None:
            self.__integrator.collision_output_file = value

    @property
    def integrator(self):
        if self.__integrator is None:
            raise RuntimeError('Integrator not set!')
        return self.__integrator

    @property
    def particles(self):
        if self.__integrator is None:
            raise RuntimeError('Particle sets undefined because the integrator is not set!')
        return self.__integrator.particles

    @property
    def t(self):
        if self.__integrator is not None:
            self.__t = self.integrator.t
            return self.__t
        else:
            return self.__t

    @property
    def CONST_G(self):
        if self.__integrator is not None:
            self.__CONST_G = self.__integrator.CONST_G
            return self.__CONST_G
        else:
            return self.__CONST_G

    @CONST_G.setter
    def CONST_G(self, value):
        self.__CONST_G = value
        if self.__integrator is not None:
            self.__integrator.CONST_G = value

    @property
    def CONST_C(self):
        if self.__integrator is not None:
            self.__CONST_C = self.__integrator.CONST_C
            return self.__CONST_C
        else:
            return self.__CONST_C

    @CONST_C.setter
    def CONST_C(self, value):
        self.__CONST_C = value
        if self.__integrator is not None:
            self.__integrator.CONST_C = value

    @property
    def t_end(self):
        if self.__integrator is not None:
            self.__t_end = self.integrator.t_end
            return self.__t_end
        else:
            return self.__t_end

    @t_end.setter
    def t_end(self, tf):
        self.__t_end = tf
        if self.__integrator is not None:
            self.integrator.t_end = tf

    @property
    def h(self):
        if self.__integrator is not None:
            self.__h = self.integrator.h
            return self.__h
        else:
            return self.__h

    @h.setter
    def h(self, value):
        self.__h = value
        if self.__integrator is not None:
            self.__integrator.h = value

    @property
    def store_dt(self):
        if self.__integrator is not None:
            self.__store_dt = self.__integrator.__store_dt
            return self.__store_dt
        else:
            return self.__store_dt

    @store_dt.setter
    def store_dt(self, value):
        self.__store_dt = value
        if self.__integrator is not None:
            self.__integrator.store_dt = self.__store_dt

    @property
    def buffer_len(self):
        if self.__integrator is not None:
            self.__buffer_len = self.__integrator.buffer_len
            return self.__buffer_len
        else:
            return self.__buffer_len

    @buffer_len.setter
    def buffer_len(self, value):
        self.__buffer_len = value
        if self.__integrator is not None:
            self.__integrator.buffer_len = self.__buffer_len

    @property
    def acceleration_method(self):
        return self.integrator.acceleration_method

    @acceleration_method.setter
    def acceleration_method(self, value):
        self.__integrator.acceleration_method = value

    @integrator.setter
    def integrator(self, name_of_integrator):
        if self.__integrators is None:
            self.__integrators = Integrator.load_integrators()
        print(('Setting the integrator to %s' % name_of_integrator))
        # populate the parameters to the integrator
        if name_of_integrator in self.__integrators:
            self.__integrator = getattr(self.__integrators[name_of_integrator], name_of_integrator)()
            self.__integrator.CONST_G = self.CONST_G
            self.__integrator.t_end = self.__t_end
            self.__integrator.h = self.__h
            self.__integrator.t_start = self.__t_start
            self.__integrator.output_file = self.output_file
            self.__integrator.collision_output_file = self.collision_output_file
            self.__integrator.close_encounter_output_file = self.close_encounter_output_file
            self.__integrator.store_dt = self.__store_dt
            self.__integrator.buffer_len = self.__buffer_len

    def initialize(self, config=None):
        # Initialize the integrator
        self.__integrators = Integrator.load_integrators()
        if self.__integrator is None:
            print('Use GaussRadau15 as the default integrator...')
            self.integrator = 'GaussRadau15'
            self.integrator.initialize()
            self.integrator.acceleration_method = 'ctypes'
        else:
            self.__integrator.CONST_G = self.CONST_G
            self.__integrator.t_end = self.__t_end
            self.__integrator.h = self.__h
            self.__integrator.t_start = self.__t_start
            self.__integrator.output_file = self.output_file
            self.__integrator.store_dt = self.__store_dt
            self.__integrator.buffer_len = self.__buffer_len


        if config is not None:
            # Gravitational parameter
            self.integrator.CONST_G = np.array(config['physical_params']['G'])

            # Integration parameters
            self.integrator = config['integration']['integrator']
            self.integrator.initialize()
            self.integrator.h = float(config['integration']['h'])
            if 'acc_method' in config['integration']:
                self.integrator.acceleration_method = config['integration']['acc_method']
            else:
                self.integrator.acceleration_method = 'ctypes'

            # Load sequence of object names
            if 'names' in config:
                names = config['names']
            else:
                names = None

            # Initial and final times
            if self.integrator.t_start == 0:
                self.integrator.t_start = float(config['integration']['t0'])
            if self.integrator.t_end == 0:
                self.integrator.t_end = float(config['integration']['tf'])
            self.integrator.active_integrator = config['integration']['integrator']
            DataIO.ic_populate(config['initial_conds'], self, names=names)

    def stop(self):
        """Stop the integrator and clean up the memory"""
        self.integrator.stop()

    def set_additional_forces(self, ext_acc):
        if ext_acc.ndim == 1 and ext_acc.shape[0] == 3 * self.integrator.particles.N:
            self.integrator.set_additional_forces(ext_acc)
        else:
            print('WARNING: Additional forces array needs to be 3 * N vector, where N is the number of particles.')

    def integrate(self, to_time=None):
        try:
            return self.integrator.integrate(to_time)
        except KeyboardInterrupt as e:
            print('Keyboard Interruption detected (Ctrl+C). Simulation stopped. Stopping the code...')
            self.stop()
            sys.exit(0)

    def calculate_orbital_elements(self, primary=None):
        return self.integrator.calculate_orbital_elements(primary)

    def calculate_energy(self):
        return self.integrator.calculate_energy()

    def add(self, pos=None, vel=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, mass=0.0, name=None,
            radius=0.0, ptype=0, a=None, e=0.0, i=0.0, Omega=0.0, omega=0.0, f=0.0, primary=None):
        if x is not None and y is not None and z is not None:
            pos = np.empty(3, dtype=np.double)
            pos[0] = x
            pos[1] = y
            pos[2] = z
        if x is not None and y is not None and z is not None:
            pos = np.empty(3, dtype=np.double)
            pos[0] = x
            pos[1] = y
            pos[2] = z
        if vx is not None and vy is not None and vz is not None:
            vel = np.empty(3, dtype=np.double)
            vel[0] = vx
            vel[1] = vy
            vel[2] = vz
        return self.integrator.particles.add(pos=pos, vel=vel, mass=mass, name=name, radius=radius,
                                             ptype=ptype, a=a, e=e, i=i, Omega=Omega,
                                             omega=omega, f=f, primary=primary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file', default=None)
    parser.add_argument('-o', '--output_file', dest='output_file', help='output data file', default='data.hdf5')
    parser.add_argument('-r', '--rebound_file', help='Rebound simulation file', default=None)
    parser.add_argument('-t', '--t_end', type=float, dest='t_end', help='Termination time')
    parser.add_argument('-d', '--dt', type=float, dest='dt', help='Integration time step (optional for certain integrators)', default=None)
    parser.add_argument('-s', '--store_dt', type=float, dest='store_dt', help='output time step', default=100)
    parser.add_argument('-i', '--integrator', dest='integrator', help='Name of the integrator [GaussRadau15|WisdomHolman|RungeKutta|AdamsBashForth|LeapFrog|Euler]', default='GaussRadau15')

    args = parser.parse_args()
    abie = ABIE()
    abie.integrator = args.integrator
    if args.output_file is not None:
        abie.output_file = args.output_file
    if args.t_end is not None:
        abie.t_end = args.t_end
    if args.config is not None:
        abie.initialize(DataIO.parse_config_file(args.config))
    elif args.rebound_file is not None:
        # populate the initial conditions from rebound simulation files
        abie.initialize()
        DataIO.ic_populate_from_rebound(args.rebound_file, abie)
    if args.dt is not None:
        abie.h = args.dt
    abie.store_dt = args.store_dt
    abie.integrate()



if __name__ == "__main__":
    main()
