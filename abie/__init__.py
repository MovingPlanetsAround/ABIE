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
from abie.particles import Particles


class ABIE(object):
    def __init__(
        self,
        CONST_G=4 * np.pi ** 2,
        CONST_C=0.0,
        integrator="GaussRadau15",
        h=0.1,
        store_dt=100,
        name="simulation",
        buffer_len=10240,
        deviceID=-1
    ):
        # =================== CONSTANTS ==================
        # by default, using the square of the Gaussian gravitational constant
        self.__CONST_G = CONST_G
        self.__CONST_C = (
            CONST_C  # speed of light; PN terms will be calculated if CONST_C > 0
        )

        # # =================== VARIABLES ==================
        self.__t = 0.0
        self.__t_start = 0.0
        self.__t_end = 0.0
        self.__h = h
        self._particles = None
        self.__buf = None
        self.__store_dt = store_dt  # output is triggered per 100 time units
        self.__buffer_len = buffer_len  # size of dataset in a hdf5 group
        self.__max_close_encounter_events = 1
        self.__max_collision_events = 1
        self.__close_encounter_distance = 0.0
        self.__device_id = deviceID

        # load integrator modules
        self.__integrator_modules = (
            None  # a collection of integrators loaded from modules
        )
        self.__integrator_instances = dict()
        self.__integrator = None  # the actual, active integrator instance
        self.output_file = "%s.hdf5" % name
        self.__close_encounter_output_file = "%s_ce.txt" % name
        self.__collision_output_file = "%s_collisions.txt" % name

    @property
    def particles(self):
        if self._particles is None:
            self._particles = Particles(self.CONST_G)
        return self._particles

    @property
    def buffer(self):
        if self.__buf is None:
            self.__buf = DataIO(
                buf_len=self.__buffer_len,
                output_file_name=self.output_file,
                CONST_G=self.CONST_G,
            )
        return self.__buf

    @property
    def max_close_encounter_events(self):
        if self.__integrator is not None:
            self.__max_close_encounter_events = (
                self.__integrator.max_close_encounter_events
            )
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
            self.__close_encounter_distance = (
                self.__integrator.max_close_encounter_events
            )
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
            self.__close_encounter_output_file = (
                self.__integrator.close_encounter_output_file
            )
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
    def t(self):
        if self.__integrator is not None:
            self.__t = self.integrator.t
            return self.__t
        else:
            return self.__t

    @property
    def CONST_G(self):
        return self.__CONST_G

    @CONST_G.setter
    def CONST_G(self, value):
        self.__CONST_G = value
        if self.__integrator is not None:
            self.__integrator.CONST_G = value

    @property
    def CONST_C(self):
        return self.__CONST_C

    @CONST_C.setter
    def CONST_C(self, value):
        self.__CONST_C = value
        if self.__integrator is not None:
            self.__integrator.CONST_C = value

    @property
    def t_end(self):
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
    def acceleration_method(self):
        return self.integrator.acceleration_method

    @acceleration_method.setter
    def acceleration_method(self, value):
        self.__integrator.acceleration_method = value

    @property
    def integrator(self):
        if self.__integrator is None:
            raise RuntimeError("Integrator not set!")
        return self.__integrator

    @integrator.setter
    def integrator(self, name_of_integrator):
        if self.__integrator_modules is None:
            self.__integrator_modules = Integrator.load_integrators()
        print(("Selecting %s as the active integrator." % name_of_integrator))
        # populate the parameters to the integrator
        if name_of_integrator in self.__integrator_modules:
            if name_of_integrator not in self.__integrator_instances:
                self.__integrator = getattr(
                    self.__integrator_modules[name_of_integrator], name_of_integrator
                )(self.particles, self.buffer, self.CONST_G, self.CONST_C, deviceID=self.__device_id)
                self.__integrator_instances[name_of_integrator] = self.__integrator
            else:
                self.__integrator = self.__integrator_instances[name_of_integrator]
            self.__integrator.CONST_G = self.CONST_G
            self.__integrator.t_end = self.__t_end
            self.__integrator.h = self.__h
            self.__integrator._t = self.__t
            self.__integrator.t_start = self.__t_start
            self.__integrator.output_file = self.output_file
            self.__integrator.collision_output_file = self.collision_output_file
            self.__integrator.close_encounter_output_file = (
                self.close_encounter_output_file
            )
        else:
            raise ValueError(
                "Unknown integrator %s. Supported integrators are %s"
                % (name_of_integrator, self.__integrator_modules)
            )

    @property
    def data(self):
        self.buffer.flush()
        return self.buffer.recorder.data

    def record_simulation(self, particles=None, quantities=None):
        if self.buffer.recorder is not None:
            self.buffer.recorder.set_monitored_particles(particles)
            self.buffer.recorder.set_monitored_quantities(quantities)
            return self.buffer.recorder
        else:
            from recorder import SimulationDataRecorder

            self.buffer.recorder = SimulationDataRecorder(particles, quantities)
            return self.buffer.recorder

    def initialize(self, config=None):
        # Initialize the integrator
        self.__integrator_modules = Integrator.load_integrators()
        if self.__integrator is None:
            print("Use GaussRadau15 as the default integrator...")
            self.integrator = "GaussRadau15"
            self.integrator.initialize()
            self.integrator.acceleration_method = "ctypes"
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
            self.integrator.CONST_G = np.array(config["physical_params"]["G"])

            # Integration parameters
            self.integrator = config["integration"]["integrator"]
            self.integrator.initialize()
            self.integrator.h = float(config["integration"]["h"])
            if "acc_method" in config["integration"]:
                self.integrator.acceleration_method = config["integration"][
                    "acc_method"
                ]
            else:
                self.integrator.acceleration_method = "ctypes"

            # Load sequence of object names
            if "names" in config:
                names = config["names"]
            else:
                names = None

            # Initial and final times
            if self.integrator.t_start == 0:
                self.integrator.t_start = float(config["integration"]["t0"])
            if self.integrator.t_end == 0:
                self.integrator.t_end = float(config["integration"]["tf"])
            self.integrator.active_integrator = config["integration"]["integrator"]
            DataIO.ic_populate(config["initial_conds"], self, names=names)

    def stop(self):
        """Stop the integrator and clean up the memory"""
        self.integrator.stop()

    def set_additional_forces(self, ext_acc):
        if ext_acc.ndim == 1 and ext_acc.shape[0] == 3 * self.integrator.particles.N:
            self.integrator.set_additional_forces(ext_acc)
        else:
            print(
                "WARNING: Additional forces array needs to be 3 * N vector, where N is the number of particles."
            )

    def integrate(self, to_time=None):
        try:
            ret = self.integrator.integrate(to_time)
            self.__t = self.integrator.t
        except KeyboardInterrupt as e:
            print(
                "Keyboard Interruption detected (Ctrl+C). Simulation stopped. Stopping the code..."
            )
            self.stop()
            sys.exit(0)

    def calculate_orbital_elements(self, primary=None):
        return self.integrator.calculate_orbital_elements(primary)

    def calculate_energy(self):
        return self.integrator.calculate_energy()

    def add(
        self,
        pos=None,
        vel=None,
        x=0.0,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        mass=0.0,
        name=None,
        radius=0.0,
        ptype=0,
        a=None,
        e=0.0,
        i=0.0,
        Omega=0.0,
        omega=0.0,
        f=2 * np.pi * np.random.rand(),
        primary=None,
    ):
        if pos is None:
            pos = np.empty(3, dtype=np.double)
            pos[0] = x
            pos[1] = y
            pos[2] = z
        if vel is None:
            vel = np.empty(3, dtype=np.double)
            vel[0] = vx
            vel[1] = vy
            vel[2] = vz
        return self.particles.add(
            pos=pos,
            vel=vel,
            mass=mass,
            name=name,
            radius=radius,
            ptype=ptype,
            a=a,
            e=e,
            i=i,
            Omega=Omega,
            omega=omega,
            f=f,
            primary=primary,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file", default=None)
    parser.add_argument(
        "-o",
        "--output_file",
        dest="output_file",
        help="output data file",
        default="data.hdf5",
    )
    parser.add_argument(
        "-r", "--rebound_file", help="Rebound simulation file", default=None
    )
    parser.add_argument(
        "-t", "--t_end", type=float, dest="t_end", help="Termination time"
    )
    parser.add_argument(
        "-d",
        "--dt",
        type=float,
        dest="dt",
        help="Integration time step (optional for certain integrators)",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--store_dt",
        type=float,
        dest="store_dt",
        help="output time step",
        default=100,
    )
    parser.add_argument(
        "-i",
        "--integrator",
        dest="integrator",
        help="Name of the integrator [GaussRadau15|WisdomHolman|RungeKutta|AdamsBashForth|LeapFrog|Euler]",
        default="GaussRadau15",
    )

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
