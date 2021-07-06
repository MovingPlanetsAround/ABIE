import ctypes
import os
import numpy as np
import sysconfig
from abie.events import CollisionException, CloseEncounterException


class CLibABIE(object):
    def __init__(self):
        __current_dir__ = os.path.dirname(os.path.realpath(__file__))
        lib_path = os.path.join(__current_dir__, "libabie.so")

        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if suffix is None:
            suffix = ".so"
        lib_path = os.path.join(__current_dir__, "..", "libabie" + suffix)
        if not os.path.isfile(lib_path):
            # try to find the library from the parent directory
            lib_path = os.path.join(
                os.path.abspath(os.path.join(__current_dir__, os.pardir)), "libabie.so"
            )

        if not os.path.isfile(lib_path):
            print("Warning! Shared library libabie.so not exsit! Trying to compile.")
            os.system("make")
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.max_close_encounter_events = 1
        self.max_collision_events = 1

    def initialize_code(
        self,
        G,
        C,
        N_MAX,
        MAX_CE_EVENTS=1,
        MAX_COLLISION_EVENTS=1,
        close_encounter_distance=0,
        deviceID=-1,
    ):
        self.lib.initialize_code(
            ctypes.c_double(G),
            ctypes.c_double(C),
            ctypes.c_int(N_MAX),
            ctypes.c_int(MAX_CE_EVENTS),
            ctypes.c_int(MAX_COLLISION_EVENTS),
            ctypes.c_int(deviceID),
        )
        self.lib.set_close_encounter_distance(ctypes.c_double(close_encounter_distance))
        self.max_close_encounter_events = MAX_CE_EVENTS
        self.max_collision_events = MAX_COLLISION_EVENTS

    def finalize_code(self):
        self.lib.finalize_code()

    def set_state(self, pos, vel, masses, radii, N, G, C):
        self.lib.set_state(
            ctypes.c_void_p(pos.ctypes.data),
            ctypes.c_void_p(vel.ctypes.data),
            ctypes.c_void_p(masses.ctypes.data),
            ctypes.c_void_p(radii.ctypes.data),
            ctypes.c_int(N),
            ctypes.c_double(G),
            ctypes.c_double(C),
        )

    def get_state(self, pos, vel, masses, radii):
        self.lib.get_state(
            ctypes.c_void_p(pos.ctypes.data),
            ctypes.c_void_p(vel.ctypes.data),
            ctypes.c_void_p(masses.ctypes.data),
            ctypes.c_void_p(radii.ctypes.data),
        )

    def get_model_time(self):
        self.lib.get_model_time.restype = ctypes.c_double
        return self.lib.get_model_time()

    def get_close_encounter_data(self):
        # each row: time, object 1, object 2, distance
        buf = np.zeros(4 * self.max_close_encounter_events)
        self.lib.get_close_encounter_buffer(ctypes.c_void_p(buf.ctypes.data))
        return buf.reshape(self.max_close_encounter_events, 4)

    def get_collision_data(self):
        # each row: time, object 1, object 2, distance
        buf = np.zeros(4 * self.max_collision_events)
        self.lib.get_collision_buffer(ctypes.c_void_p(buf.ctypes.data))
        return buf.reshape(self.max_collision_events, 4)

    def reset_close_encounter_buffer(self):
        self.lib.reset_close_encounter_buffer()

    def reset_collision_buffer(self):
        self.lib.reset_collision_buffer()

    def set_close_encounter_distance(self, value):
        self.lib.set_close_encounter_distance(ctypes.c_double(value))

    def get_total_energy(self):
        self.lib.calculate_energy.restype = ctypes.c_double
        return self.lib.calculate_energy()

    def set_additional_forces(self, ext_acc):
        """
        :param ext_acc: A 3 * N vector
        :return:
        """
        self.lib.set_additional_forces(
            ctypes.c_int(len(ext_acc) / 3), ctypes.c_void_p(ext_acc.ctypes.data)
        )

    def integrator_runge_kutta(self, pos, vel, masses, N, G, t, t_end, dt):
        self.lib.integrator_runge_kutta(
            ctypes.c_void_p(pos.ctypes.data),
            ctypes.c_void_p(vel.ctypes.data),
            ctypes.c_void_p(masses.ctypes.data),
            ctypes.c_int(N),
            ctypes.c_double(G),
            ctypes.c_double(t),
            ctypes.c_double(t_end),
            ctypes.c_double(dt),
        )

    def integrator_gauss_radau15(self, pos, vel, masses, N, G, t, t_end, dt):
        self.lib.integrator_gauss_radau15(
            ctypes.c_void_p(pos.ctypes.data),
            ctypes.c_void_p(vel.ctypes.data),
            ctypes.c_void_p(masses.ctypes.data),
            ctypes.c_int(N),
            ctypes.c_double(G),
            ctypes.c_double(t),
            ctypes.c_double(t_end),
            ctypes.c_double(dt),
        )

    def integrator_wisdom_holman(self, pos, vel, masses, N, G, t, t_end, dt):
        self.lib.integrator_wisdom_holman(
            ctypes.c_void_p(pos.ctypes.data),
            ctypes.c_void_p(vel.ctypes.data),
            ctypes.c_void_p(masses.ctypes.data),
            ctypes.c_int(N),
            ctypes.c_double(G),
            ctypes.c_double(t),
            ctypes.c_double(t_end),
            ctypes.c_double(dt),
        )

    def integrator_gr(self, t, t_end, dt):
        self.lib.integrator_gr.restype = ctypes.c_int
        ret = self.lib.integrator_gr(
            ctypes.c_double(t), ctypes.c_double(t_end), ctypes.c_double(dt)
        )
        if ret == 1:
            col_buf = self.get_close_encounter_data()
            raise CloseEncounterException(
                col_buf[-1, 0], int(col_buf[-1, 1]), int(col_buf[-1, 2]), col_buf[-1, 3]
            )
        elif ret == 2:
            col_buf = self.get_collision_data()
            raise CollisionException(
                col_buf[-1, 0], int(col_buf[-1, 1]), int(col_buf[-1, 2]), col_buf[-1, 3]
            )

    def integrator_rk(self, t, t_end, dt):
        self.lib.integrator_rk(
            ctypes.c_double(t), ctypes.c_double(t_end), ctypes.c_double(dt)
        )

    def integrator_wh(self, t, t_end, dt):
        self.lib.integrator_wh(
            ctypes.c_double(t), ctypes.c_double(t_end), ctypes.c_double(dt)
        )

    def ode_n_body_first_order(self, x, N, G, masses):
        dxdt = np.zeros(x.size, dtype=np.double)
        self.lib.ode_n_body_first_order(
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_int(N),
            ctypes.c_double(G),
            ctypes.c_void_p(masses.ctypes.data),
            ctypes.c_void_p(dxdt.ctypes.data),
        )
        return dxdt

    def ode_n_body_second_order(self, x, N, G, masses):
        acc = np.zeros(x.size, dtype=np.double)
        self.lib.ode_n_body_second_order(
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_int(N),
            ctypes.c_double(G),
            ctypes.c_void_p(masses.ctypes.data),
            ctypes.c_void_p(acc.ctypes.data),
        )
        return acc
