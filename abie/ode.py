import numpy as np
import ctypes
import os

# from clibabie import CLibABIE

# if not os.path.isfile('libabie.so'):
#     print('Warning! Shared library libabie.so not exsit! Trying to compile.')
#     os.system('make')
# lib = ctypes.cdll.LoadLibrary("./libabie.so")
# fun_ode_first_order = lib.ode_n_body_first_order
# fun_ode_second_order = lib.ode_n_body_second_order
# fun.restype = None
# fun.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
#                 ctypes.c_size_t,
#                 ctypes.c_double,
#                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
#                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


# libabie = CLibABIE()


class ODE(object):

    # @staticmethod
    # def ode_n_body_second_order(x, const_g, masses):
    #     dxdt = libabie.ode_n_body_second_order(x, x.size/3, const_g, masses)
    #     return dxdt

    @staticmethod
    def ode_n_body_first_order(x, const_g, masses):
        """
        TWO-BODY EQUATIONS
        :param x: the flattened state vector. The first half of x is the positions; the second half is velocities.
        :param const_g:
        :param masses:
        :return: the first derivative of x; the first half is velocities; the second half is accelerations.
        """
        # Allocate
        dxdt = x * 0.0

        # Compute additional perturbations
        # perturbation = compute_perturbation(x, t, G, masses, nbodies)

        # Differential equations:
        # - Position
        nbodies = x.size // 6
        dxdt[0 : nbodies * 3] = x[nbodies * 3 :]  # velocities
        for j in range(0, nbodies):
            # dxdt[j * 3:3 + j * 3] = x[nbodies * 3 + j * 3:nbodies * 3 + 3 + j * 3]
            Rj = x[j * 3 : 3 + j * 3]
            aj = Rj * 0
            for k in range(0, nbodies):
                if j == k:
                    continue
                Rk = x[k * 3 : k * 3 + 3]
                rel_sep = Rj - Rk
                aj += -const_g * masses[k] * rel_sep / np.linalg.norm(rel_sep) ** 3
            dxdt[nbodies * 3 + j * 3 : nbodies * 3 + 3 + j * 3] = aj  # accelerations

        # Add extra accelerations
        # dxdt[nbodies * 3:] += perturbation
        return dxdt

    @staticmethod
    def ode_n_body_second_order(x, const_g, masses):

        nbodies = x.size / 3  # WARNING: this x contains only positions!!

        # Allocate
        acc = x * 0.0

        # Compute additional perturbations
        # perturbation = compute_perturbation(x, t, G, masses, nbodies)

        # Differential equations:
        # - Position
        for j in range(0, nbodies):
            Rj = x[j * 3 : 3 + j * 3]
            aj = Rj * 0
            for k in range(0, nbodies):
                if j == k:
                    continue
                Rk = x[k * 3 : 3 + k * 3]
                rel_sep = Rj - Rk
                aj += -const_g * masses[k] * rel_sep / np.linalg.norm(rel_sep) ** 3
            acc[j * 3 : 3 + j * 3] = aj

        # Add extra accelerations
        # acc += perturbation

        return acc
