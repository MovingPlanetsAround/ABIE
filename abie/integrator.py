"""
Alice-Bob Integrator (ABI), created by Alice and Bob in the Moving Planets Around (MPA) project.

Features:
    1. The integrator takes input either from the commandline, or from the config file
    2. The integrator periodically stores data to HDF5
    3. The integrator supports restarting simulations


The MPA team, 2017-2018
"""

import glob
import os
import sys
import numpy as np
from abie.particles import Particles
from abie.clibabie import CLibABIE
from abie.data_io import DataIO

__mpa_dir__ = os.path.dirname(os.path.abspath(__file__))
__user_shell_dir__ = os.getcwd()


class Integrator(object):
    def __init__(
        self, particles=None, buffer=None, CONST_G=4 * np.pi ** 2, CONST_C=0.0, buf_len=10240, deviceID=-1
    ):
        """
        The constructor of an abstract integrator
        """
        # =================== CONSTANTS ==================
        self.CONST_G = CONST_G
        self.CONST_C = (
            CONST_C  # speed of light; PN terms will be calculated if CONST_C > 0
        )

        # =================== VARIABLES ==================
        self._t = 0.0
        self.t_start = 0.0
        self.t_end = 0.0
        self.h = 0.01  # time step size
        self.store_dt = 100  # storage time step
        self._particles = particles
        self.acceleration_method = "ctypes"
        self.output_file = "data.hdf5"
        self.collision_output_file = "collisions.txt"
        self.close_encounter_output_file = "close_encounters.txt"
        self.max_close_encounter_events = 1
        self.max_collision_events = 1
        self.close_encounter_distance = 0.0
        self.__energy_init = 0.0
        self.__energy = 0.0
        self.__buf = buffer
        self.buffer_len = buf_len
        self.__initialized = False
        self.__device_id = deviceID
        print('DevID', deviceID)


        # =============== C Library =============
        self.libabie = CLibABIE()

    @property
    def t(self):
        return self._t

    @property
    def particles(self):
        if self._particles is None:
            self._particles = Particles(self.CONST_G)
        return self._particles

    @property
    def buf(self):
        if self.__buf is None:
            self.__buf = DataIO(
                buf_len=self.buffer_len,
                output_file_name=self.output_file,
                CONST_G=self.CONST_G,
            )
        return self.__buf

    @staticmethod
    def load_integrators():
        """
        Load integrator modules
        :return: a dict of integrator class objects, mapping the name of the integrator to the class object
        """
        mod_dict = dict()
        module_candidates = glob.glob(os.path.join(__mpa_dir__, "integrator_*.py"))
        sys.path.append(__mpa_dir__)  # append the python path
        if __mpa_dir__ != __user_shell_dir__:
            # load the integrator module (if any) also from the current user shell directory
            module_cwd = glob.glob(os.path.join(__user_shell_dir__, "integrator_*.py"))
            for m_cwd in module_cwd:
                module_candidates.append(m_cwd)
            sys.path.append(__user_shell_dir__)  # append the python path
        for mod_name in module_candidates:
            mod_name = os.path.basename(mod_name)
            mod = __import__(mod_name.split(".")[0])
            if hasattr(mod, "__integrator__"):
                # it is a valid ABI module, register it as a module
                mod_dict[mod.__integrator__] = mod
        return mod_dict

    def initialize(self):
        if self.__buf is None:
            self.__buf = DataIO(
                buf_len=self.buffer_len,
                output_file_name=self.output_file,
                close_encounter_output_file_name=self.close_encounter_output_file,
                collision_output_file_name=self.collision_output_file,
                CONST_G=self.CONST_G,
            )
        if self.particles.N > 0:
            # initialize the C library
            self.libabie.initialize_code(
                self.CONST_G,
                self.CONST_C,
                self.particles.N,
                MAX_CE_EVENTS=self.max_close_encounter_events,
                MAX_COLLISION_EVENTS=self.max_collision_events,
                close_encounter_distance=self.close_encounter_distance,
                deviceID=self.__device_id
            )
            self.buf.initialize_buffer(self.particles.N)

    def stop(self):
        if self.__buf is not None:
            self.__buf.close()

    def calculate_orbital_elements(self, primary=None):
        return self._particles.calculate_orbital_elements(primary)

    def calculate_energy(self):
        # return self._particles.energy
        if self.acceleration_method == "ctypes":
            return self.libabie.get_total_energy()
        else:
            return self._particles.energy

    def set_additional_forces(self, ext_acc):
        self.libabie.set_additional_forces(ext_acc)

    def integrator_warmup(self):
        pos = self.particles.positions.copy()
        vel = self.particles.velocities.copy()
        self.libabie.set_state(
            pos,
            vel,
            self.particles.masses,
            self.particles.radii,
            self.particles.N,
            self.CONST_G,
            self.CONST_C,
        )

    def integrate(self, to_time=None):
        """
        Integrate the system to a given time.
        :param to_time: The termination time. If None, it will use the self.t_end value, and the code will be stopped
                        when reaching self.t_end (i.e. if this function is called without argument, it can only be called
                        once; but if it is called with a to_time argument specificed, then it can be called iteratively.
        :return:
        """
        if to_time is not None:
            self.t_end = to_time

        if self.__initialized is False:
            self.initialize()
            self.integrator_warmup()
            self.__initialized = True
        if self.t >= self.t_end:
            return

        # Note: this dt is not the integrator time step h
        dt = min(self.store_dt, self.t_end - self.t)

        ret = 0
        # launch the integration
        while self.t < self.t_end:
            if self.__energy_init == 0:
                self.__energy_init = self.calculate_energy()
            next_t = self.t + dt - ((self.t + dt) % dt)
            if self.acceleration_method == "numpy":
                ret = self.integrate_numpy(next_t)
            elif self.acceleration_method == "ctypes":
                ret = self.integrate_ctypes(next_t)
            # the self.t is updated by the subclass
            # energy check
            self.__energy = self.calculate_energy()
            print(
                (
                    "t = %f, N = %d, dE/E0 = %g"
                    % (
                        self.t,
                        self.particles.N,
                        np.abs(self.__energy - self.__energy_init) / self.__energy_init,
                    )
                )
            )
            if os.path.isfile("STOP"):
                break

        if to_time is None:
            # triggering the termination of the code, save the buffer to the file and close it
            self.stop()
            # if ret > 0:
            #     break
        # if self.t == self.t_end:
        #     self.__energy = self.calculate_energy()
        #     print('t = %f, E/E0 = %g' % (self.t, np.abs(self.__energy - self.__energy_init) / self.__energy_init))
        return ret

    def integrate_numpy(self, to_time):
        """
        Integrate the system to a given time using python/numpy.
        This method must be implemented in the subclasses.
        :param to_time: The termination time. If None, it will use the self.t_end value
        :return:
        """
        raise NotImplementedError("integrate_numpy() method not implemented!")

    def integrate_ctypes(self, to_time):
        """
        Integrate the system to a given time using the ctypes (libabie.so)
        This method must be implemented in the subclasses.
        :param to_time: The termination time. If None, it will use the self.t_end value
        :return:
        """
        raise NotImplementedError("integrate_ctypes() method not implemented!")

    def store_state(self):
        if self.buf is None:
            self.initialize()
        self.buf.initialize_buffer(self.particles.N)
        elem = self.particles.calculate_aei()
        if self.__energy == 0:
            self.__energy = self.calculate_energy()
        self.buf.store_state(
            self.t,
            self.particles.positions,
            self.particles.velocities,
            self.particles.masses,
            radii=self.particles.radii,
            names=self.particles.hashes,
            ptypes=self.particles.ptypes,
            a=elem[:, 0],
            e=elem[:, 1],
            i=elem[:, 2],
            energy=self.__energy,
        )

    def store_collisions(self, collision_buffer):
        self.buf.store_collisions(collision_buffer)

    def store_close_encounters(self, ce_buffer):
        self.buf.store_close_encounters(ce_buffer)

    def handle_collisions(self, collision_buffer, actions=None):
        if actions is None:
            actions = ["merge", "store"]
        if "store" in actions:
            self.store_state()
            self.store_collisions(collision_buffer)
        if "merge" in actions:
            collision_buffer = collision_buffer.reshape(len(collision_buffer), 4)
            for coll_pair in range(collision_buffer.shape[0]):
                pid1 = int(collision_buffer[coll_pair, 1])
                pid2 = int(collision_buffer[coll_pair, 2])
                self.particles.merge_particles_inelastically(pid1, pid2)
                self.libabie.reset_collision_buffer()
                self.integrator_warmup()
                self.buf.flush()
                self.buf.reset_buffer()
                self.buf.initialize_buffer(self.particles.N)
        if "halt" in actions:
            print("Simulation terminated due to a collision event.")
            sys.exit(0)

    def handle_close_encounters(self, ce_buffer, actions=None):
        if actions is None:
            actions = ["merge", "store"]
        if "store" in actions:
            self.store_state()
            self.store_close_encounters(ce_buffer)
        if "halt" in actions:
            print("Simulation terminated due to a collision event.")
            sys.exit(0)
