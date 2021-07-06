from abie.integrator import Integrator
from abie.ode import ODE
import numpy as np
import ctypes
import os

__integrator__ = "RungeKutta"

# if not os.path.isfile('libode.so'):
#     print('Warning! Shared library libode.so not exsit! Trying to compile.')
#     os.system('cc -fPIC -shared -o libode.so ode.c integrator_gauss_radau15.c -O3')
# lib = ctypes.cdll.LoadLibrary("./libode.so")
# fun = lib.runge_kutta


class RungeKutta(Integrator):
    def __init__(
        self, particles=None, buffer=None, CONST_G=4 * np.pi ** 2, CONST_C=0.0, deviceID=-1
    ):
        super(self.__class__, self).__init__(particles=particles, 
                                             buffer=buffer, 
                                             CONST_G=CONST_G, 
                                             CONST_C=CONST_C, 
                                             deviceID=deviceID)

    def integrate_ctypes(self, to_time=None):
        energy_init = self.calculate_energy()
        dt = min(self.store_dt, self.t_end - self.t)
        # self.libabie.initialize_code(self.CONST_G, self.particles.N)
        pos = self.particles.positions.copy()
        vel = self.particles.velocities.copy()
        self.libabie.set_state(
            pos,
            vel,
            self.particles.masses,
            self.particles.radii,
            self.particles.N,
            self.CONST_G,
        )
        energy = self.calculate_energy()
        print(
            ("t = %f, E/E0 = %g" % (self.t, np.abs(energy - energy_init) / energy_init))
        )
        self.store_state()
        if to_time is not None:
            self.t_end = to_time

        if self.h == 0.0:
            print("ERROR: the timestep for RungeKutta is not set!!! Exiting...")
            import sys

            sys.exit(0)

        while self.t < self.t_end:
            # pos = self.particles.positions.copy()
            # vel = self.particles.velocities.copy()
            # self.libabie.integrator_gauss_radau15(pos, vel, self.particles.masses, self.particles.N, self.CONST_G, self.t, self.t+dt, dt)
            self.libabie.integrator_rk(self.t, self.t + dt, self.h)
            self._t += dt
            self.libabie.get_state(
                pos, vel, self.particles.masses, self.particles.radii
            )
            self.particles.positions = pos
            self.particles.velocities = vel
            self.store_state()
            energy = self.calculate_energy()
            print(
                (
                    "t = %f, E/E0 = %g"
                    % (self.t, np.abs(energy - energy_init) / energy_init)
                )
            )
        self.buf.close()
        return 0

        # dt = min(self.store_dt, self.t_end-self.t)
        # energy_init = self.calculate_energy()
        # self.libabie.initialize_code(self.CONST_G, self.particles.N)
        # if to_time is not None:
        #     self.t_end = to_time
        # while self.t < self.t_end:
        #     pos = self.particles.positions.copy()
        #     vel = self.particles.velocities.copy()
        #     self.libabie.integrator_runge_kutta(pos, vel, self.particles.masses, self.particles.N, self.CONST_G, self.t, self.t+dt, self.h)
        #     self.particles.positions = pos
        #     self.particles.velocities = vel
        #     self._t += dt
        #     self.store_state()
        #     energy = self.calculate_energy()
        #     print('t = %f, E/E0 = %g' % (self.t, np.abs(energy-energy_init)/energy_init))

        # # Allocate dense output
        # npts = int(np.floor((self.t_end - self.t_start) / self.h) + 1)
        #
        # # Initial state
        # x = np.concatenate((self._particles.positions, self._particles.velocities))
        # # Vector of times
        # sol_time = np.linspace(self.t_start, self.t_start + self.h * (npts - 1), npts)
        # energy_init = self.calculate_energy()
        #
        # dxdt = np.zeros(x.size, dtype=np.double)
        # # Launch integration
        # count = 1
        # for t in sol_time[count:]:
        #     fun(ctypes.c_void_p(x.ctypes.data),
        #         ctypes.c_int(x.size / 6),
        #         ctypes.c_double(self.CONST_G),
        #         ctypes.c_double(self.h),
        #         ctypes.c_void_p(self.particles.masses.ctypes.data))
        #     # Store step
        #     self.particles.positions = x[0:self._particles.N * 3]
        #     self.particles.velocities = x[self._particles.N * 3:]
        #     self._t = t
        #     self.store_state()
        #     energy = self.calculate_energy()
        #     print('t = %f, E/E0 = %g' % (self.t, np.abs(energy - energy_init) / energy_init))
        #     count += 1

    def integrate_numpy(self, to_time=None):
        if to_time is not None:
            self.t_end = to_time
        # Allocate dense output
        npts = int(np.floor((self.t_end - self.t_start) / self.h) + 1)

        # Initial state
        x = np.concatenate((self._particles.positions, self._particles.velocities))
        # Vector of times
        sol_time = np.linspace(self.t_start, self.t_start + self.h * (npts - 1), npts)
        energy_init = self.calculate_energy()
        # Launch integration
        count = 1
        for t in sol_time[count:]:
            # Evaluate coefficients
            k1 = ODE.ode_n_body_first_order(x, self.CONST_G, self._particles.masses)
            k2 = ODE.ode_n_body_first_order(
                x + 0.5 * self.h * k1, self.CONST_G, self._particles.masses
            )
            k3 = ODE.ode_n_body_first_order(
                x + 0.5 * self.h * k2, self.CONST_G, self._particles.masses
            )
            k4 = ODE.ode_n_body_first_order(
                x + self.h * k3, self.CONST_G, self._particles.masses
            )

            # Advance the state
            x += self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            # Store step
            self.particles.positions = x[0 : self._particles.N * 3]
            self.particles.velocities = x[self._particles.N * 3 :]
            self._t = t
            self.store_state()
            energy = self.calculate_energy()
            print(
                (
                    "t = %f, E/E0 = %g"
                    % (self.t, np.abs(energy - energy_init) / energy_init)
                )
            )
            count += 1
        self.buf.close()
        return 0
