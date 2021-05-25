"""
Simulation data recorder.

Maxwell X. Cai, April 2021.

"""
import numpy as np


class SimulationDataRecorder(object):
    def __init__(self, particles=None, quantities=["a", "ecc", "inc"], buffer_len=0):

        self._monitored_particles = particles
        self._monitored_quantities = quantities
        self._recording = False
        self._particles_hash_vec = None

        # initialize storage
        self.data = dict()

    def reset(self):
        del self.data
        self.data = dict()

    def start(self):
        self._recording = True

    def stop(self):
        self._recording = False

    def set_monitored_quantities(self, quantities):
        self._monitored_quantities = quantities

    def get_monitored_quantities(self, quantities):
        return self._monitored_quantities

    def set_monitored_particles(self, particles):
        self._monitored_particles = particles

    def get_monitored_particles(self, particles):
        self._monitored_particles = particles

    def record(self, state_dict):
        if self._recording is False:
            return

        if self._monitored_quantities is None:
            quantities = [
                "a",
                "ecc",
                "inc",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "mass",
                "time",
                "energy",
            ]
        else:
            quantities = self._monitored_quantities

        for i, qty in enumerate(quantities):
            if qty in state_dict:
                d = state_dict[qty]
                if d.ndim == 2:
                    d = d[
                        ~np.isnan(d).any(axis=1)
                    ]  # remove the parts of buffer that contains NaN
                else:
                    d = d[~np.isnan(d)]

                if qty in self.data and self.data[qty] is not None:
                    # append
                    indices = np.where(
                        np.in1d(self._particles_hash_vec, state_dict["hash"][0])
                    )[
                        0
                    ]  # the indices to be updated
                    if self._monitored_particles is not None:
                        if d.ndim == 2:
                            tmp_data = (
                                np.zeros((d.shape[0], self.data[qty].shape[1])) * np.nan
                            )
                            tmp_data[:, indices] = d[self._monitored_particles]
                            # print(tmp_data, indices, self._particles_hash_vec, state_dict['hash'][0])
                            self.data[qty] = np.append(self.data[qty], tmp_data, axis=0)
                        else:
                            self.data[qty] = np.append(
                                self.data[qty], d[self._monitored_particles]
                            )
                    else:
                        if d.ndim == 2:
                            tmp_data = (
                                np.zeros((d.shape[0], self.data[qty].shape[1])) * np.nan
                            )
                            tmp_data[:, indices] = d
                            # print(tmp_data, indices, self._particles_hash_vec, state_dict['hash'][0])
                            self.data[qty] = np.append(self.data[qty], tmp_data, axis=0)
                        else:
                            self.data[qty] = np.append(self.data[qty], d)
                else:
                    # New record
                    self._particles_hash_vec = state_dict["hash"][
                        0
                    ]  # get the hash of all particles at t = 0
                    if self._monitored_particles is not None:
                        self.data[qty] = d[self._monitored_particles].copy()
                    else:
                        self.data[qty] = d.copy()
