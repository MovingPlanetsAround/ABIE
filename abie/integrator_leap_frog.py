from abie.integrator import Integrator
from abie.ode import ODE
import numpy as np

__integrator__ = "LeapFrog"


class LeapFrog(Integrator):
    def __init__(
        self, particles=None, buffer=None, CONST_G=4 * np.pi ** 2, CONST_C=0.0, deviceID=-1
    ):
        super(LeapFrog, self).__init__(particles=particles, 
                                       buffer=buffer, 
                                       CONST_G=CONST_G, 
                                       CONST_C=CONST_C, 
                                       deviceID=deviceID)
        self.__initialized = False

    def integrate(self, to_time=None):
        if self.__initialized is False:
            self.initialize()
            self.__initialized = True
        print(to_time, self.t_end)
        if to_time is not None:
            self.t_end = to_time
        # Allocate dense output
        npts = int(np.floor((self.t_end - self.t_start) / self.h) + 1)

        # Initial state
        x = np.concatenate((self.particles.positions, self.particles.velocities))
        # Vector of times
        sol_time = np.linspace(self.t_start, self.t_start + self.h * (npts - 1), npts)
        energy_init = self.calculate_energy()
        # Compute second step
        dxdt0 = ODE.ode_n_body_first_order(x, self.CONST_G, self.particles.masses)
        x = x + dxdt0 * self.h

        # Launch integration
        count = 2
        print("sol_time", sol_time)
        for t in sol_time[count:]:
            dxdt = ODE.ode_n_body_first_order(x, self.CONST_G, self.particles.masses)
            # Advance step
            x += 0.5 * self.h * (3 * dxdt - dxdt0)

            # Update
            dxdt0 = dxdt
            self.particles.positions = x[0 : self.particles.N * 3]
            self.particles.velocities = x[self.particles.N * 3 :]
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
