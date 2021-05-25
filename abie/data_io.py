import toml
import numpy as np
import h5py
import os
from abie.recorder import SimulationDataRecorder


class DataIO(object):
    def __init__(
        self,
        buf_len=1024,
        output_file_name="data.hdf5",
        collision_output_file_name="collisions.txt",
        close_encounter_output_file_name="close_encounters.txt",
        CONST_G=1,
    ):
        self.recorder = SimulationDataRecorder()
        self.buf_len = buf_len
        self.buf_initialized = False
        self.buf_t = None  # for the time vector
        self.store_t = 0.0  # the current time of the snapshot data in the buffer
        self.buf_energy = None  # store the total energy
        self.buf_state = None  # for x, y, z, vx, vy, vz
        self.buf_x = None
        self.buf_y = None
        self.buf_z = None
        self.buf_vx = None
        self.buf_vy = None
        self.buf_vz = None
        self.buf_mass = None
        self.buf_radius = None
        self.buf_hashes = None
        self.buf_ptype = None
        self.buf_semi = None
        self.buf_ecc = None
        self.buf_inc = None
        self.buf_cursor = 0
        self.output_file_name = output_file_name
        self.collision_output_file_name = collision_output_file_name
        self.close_encounter_output_file_name = close_encounter_output_file_name
        self.h5_file = None
        self.h5_step_id = 0
        self.CONST_G = CONST_G

    def initialize_buffer(self, n_particles):
        if self.buf_initialized is False:
            buf_len = self.buf_len
            self.buf_t = np.zeros(buf_len) * np.nan
            self.buf_energy = np.zeros(buf_len) * np.nan
            self.buf_mass = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_ptype = np.zeros((buf_len, n_particles), dtype=np.int) * np.nan
            self.buf_hashes = np.zeros((buf_len, n_particles), dtype=np.int) * np.nan
            self.buf_radius = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_x = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_y = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_z = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_vx = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_vy = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_vz = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_semi = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_ecc = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_inc = np.zeros((buf_len, n_particles)) * np.nan
            self.buf_cursor = 0
            # self.h5_step_id = 0
            # remove the previously generated collision / close encounter files
            if os.path.isfile(self.close_encounter_output_file_name):
                os.remove(self.close_encounter_output_file_name)
            if os.path.isfile(self.collision_output_file_name):
                os.remove(self.collision_output_file_name)

            self.buf_initialized = True

    def reset_buffer(self):
        self.buf_initialized = False

    def flush(self):
        """
        Write the simulation data buffer to the HDF5 file unconditionally.
        :return:
        """
        if self.buf_cursor == 0:
            # already flushed
            return

        if self.h5_file is None:
            self.h5_file = h5py.File(self.output_file_name, "w")
            self.h5_file.attrs["G"] = self.CONST_G
        h5_step_group = self.h5_file.create_group("Step#%d" % self.h5_step_id)

        state_dict = {
            "time": self.buf_t[: self.buf_cursor],
            "mass": self.buf_mass[: self.buf_cursor],
            "ptype": self.buf_ptype[: self.buf_cursor],
            "hash": self.buf_hashes[: self.buf_cursor],
            "radius": self.buf_radius[: self.buf_cursor],
            "x": self.buf_x[: self.buf_cursor],
            "y": self.buf_y[: self.buf_cursor],
            "z": self.buf_z[: self.buf_cursor],
            "vx": self.buf_vx[: self.buf_cursor],
            "vy": self.buf_vy[: self.buf_cursor],
            "vz": self.buf_vz[: self.buf_cursor],
            "a": self.buf_semi[: self.buf_cursor],
            "ecc": self.buf_ecc[: self.buf_cursor],
            "inc": self.buf_inc[: self.buf_cursor],
            "energy": self.buf_energy[: self.buf_cursor],
        }
        for dset in state_dict.keys():
            h5_step_group.create_dataset(dset, data=state_dict[dset])
        self.h5_file.flush()

        # reset the cursor
        self.h5_step_id += 1
        self.buf_cursor = 0

        # record
        self.recorder.record(state_dict=state_dict)

    def close(self):
        if self.buf_cursor > 0:
            # there are still some buffer data to be written to the HDF5 file
            self.flush()
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def store_state(
        self,
        t,
        pos,
        vel,
        masses,
        radii=None,
        names=None,
        ptypes=None,
        a=None,
        e=None,
        i=None,
        energy=None,
    ):
        # return if the snapshot time is the same as store_t, which means that this time is already stored
        if self.store_t == t:
            return

        if self.buf_cursor == self.buf_len:
            # the buffer is full, trigger store
            self.flush()
            self.buf_cursor = 0

        self.buf_t[self.buf_cursor] = t
        self.buf_x[self.buf_cursor] = pos[0::3]  # [0, 3, 6, ...]
        self.buf_y[self.buf_cursor] = pos[1::3]  # [1, 4, 7, ...]
        self.buf_z[self.buf_cursor] = pos[2::3]  # [2, 5, 8, ...]
        self.buf_vx[self.buf_cursor] = vel[0::3]  # [0, 3, 6, ...]
        self.buf_vy[self.buf_cursor] = vel[1::3]  # [1, 4, 7, ...]
        self.buf_vz[self.buf_cursor] = vel[2::3]  # [2, 5, 8, ...]
        self.buf_mass[self.buf_cursor] = masses
        if radii is not None:
            self.buf_radius[self.buf_cursor] = radii
        if names is not None:
            self.buf_hashes[self.buf_cursor] = names
        if ptypes is not None:
            self.buf_ptype[self.buf_cursor] = ptypes
        if a is not None:
            self.buf_semi[self.buf_cursor] = a
        if e is not None:
            self.buf_ecc[self.buf_cursor] = e
        if i is not None:
            self.buf_inc[self.buf_cursor] = i
        if energy is not None:
            self.buf_energy[self.buf_cursor] = energy

        self.store_t = t
        self.buf_cursor += 1

    def store(self, buf_t, buf_state, buf_len, n_particles):
        """
        Append `buf_t` and `buf_state` to the instance-wide simulation data buffer. If the buffer is full, it will be
        written to the HDF5 file and then emptied.

        :param buf_t:
        :param buf_state:
        :param buf_len:
        :param n_particles:
        :return:
        """
        if self.buf_len - self.buf_cursor >= buf_len:
            # the buffer storage is sufficient
            self.buf_t[self.buf_cursor : self.buf_cursor + buf_len] = buf_t[:buf_len]
            self.buf_state[self.buf_cursor : self.buf_cursor + buf_len] = buf_state[
                :buf_len
            ]
            self.buf_cursor += buf_len
        else:
            # full the buffer until it is full, leave the remaining part for the next output
            remaining_buf_len = self.buf_cursor - self.buf_cursor
            self.buf_t[self.buf_cursor : self.buf_len] = buf_t[:remaining_buf_len]
            self.buf_state[self.buf_cursor : self.buf_len] = buf_state[
                :remaining_buf_len
            ]
            self.buf_cursor = self.buf_len

            # trigger file output if the buffer is full
            self.flush()
            self.buf_cursor = 0

            # store the remaining part into the emptied buffer
            self.buf_t[0 : buf_len - remaining_buf_len] = buf_t[
                remaining_buf_len:buf_len
            ]
            self.buf_state[0 : buf_len - remaining_buf_len] = buf_state[
                remaining_buf_len:buf_len
            ]

        # trigger file output if the buffer is full
        if self.buf_cursor == self.buf_len:
            self.flush()
            self.buf_cursor = 0
            # adjust the buffer length if the default value is too small
            if self.buf_len < buf_len:
                self.buf_len = buf_len

    def store_collisions(self, collision_buffer):
        if self.collision_output_file_name is not None:
            np.savetxt(
                self.collision_output_file_name,
                collision_buffer,
                fmt="%g, %d, %d, %g",
                header="Time, Particle 1, Particle 2, Distance",
            )

    def store_close_encounters(self, ce_buffer):
        if self.close_encounter_output_file_name is not None:
            np.savetxt(
                self.close_encounter_output_file_name,
                ce_buffer,
                fmt="%g, %d, %d, %g",
                header="Time, Particle 1, Particle 2, Distance",
            )

    @staticmethod
    def parse_config_file(config_file):
        with open(config_file) as conf_file:
            config = toml.loads(conf_file.read())
            return config  # this is a dict structure

    @staticmethod
    def ic_populate_from_rebound(reb_ic_filename, sim_abie):
        try:
            import rebound

            sim_reb = rebound.Simulation().from_file(reb_ic_filename)
            sim_abie.CONST_G = sim_reb.G
            sim_abie.h = sim_reb.dt
            for particle_id in range(sim_reb.N):
                p = sim_reb.particles[particle_id]
                sim_abie.add(
                    mass=p.m,
                    x=p.x,
                    y=p.y,
                    z=p.z,
                    vx=p.vx,
                    vy=p.vy,
                    vz=p.vz,
                    name=p.hash.value,
                    radius=p.r,
                )
            # data = dict()
            #
            # sim_reb = rebound.Simulation().from_file(reb_ic_filename)
            # data_phy = dict()  # physical parameters
            # data_phy['G'] = sim_reb.G
            # print sim_reb.G
            # data_int = dict()  # integrator
            # data_int['t0'] = 0.0
            # data_int['tf'] = 0.0
            # data_int['h'] = sim_reb.dt
            # data_int['integrator'] = 'GaussRadau15'
            # data_int['acc_method'] = 'ctypes'
            #
            # data_ic = dict()  # initial conditions
            # names = []
            # for particle_id in range(sim_reb.N):
            #     p = sim_reb.particles[particle_id]
            #     pos = [p.x, p.y, p.z]
            #     vel = [p.vx, p.vy, p.vz]
            #     isv = np.append(pos, vel)
            #     isv = np.append(isv, p.m)
            #     p_name = 'p_%05d' % particle_id
            #     data_ic[p_name] = isv.tolist()
            #     names.append(p_name)
            # # sort to preserve the order
            # data['physical_params'] = data_phy
            # data['integration'] = data_int
            # data['initial_conds'] = data_ic
            # data['names'] = names
            #
            # toml.dump(data, open('ss_reb.conf', 'w'))
            # return data
        except ImportError:
            print("Rebound is not installed. Cannot populate IC from rebound.")
            return None

    @staticmethod
    def ic_populate(ic_dict, sim, names=None):
        if names is not None:
            # add the object preserving the name order
            for p_name in names:
                pdata = np.array(ic_dict[p_name])
                sim.add(pos=pdata[0:3], vel=pdata[3:6], mass=pdata[6], name=p_name)
        else:
            # add the objects with random order
            for planet in ic_dict:
                pdata = np.array(ic_dict[planet])
                sim.add(pos=pdata[0:3], vel=pdata[3:6], mass=pdata[6], name=planet)

    @staticmethod
    def toml_generator(state_vec, G, t0, tf, h, masses, filename):
        data = dict()
        data_phy = dict()
        data_phy["G"] = G

        data_int = dict()
        data_int["t0"] = t0
        data_int["tf"] = tf
        data_int["h"] = h

        data_ic = dict()
        ss = [
            "Sun",
            "Mercury",
            "Venus",
            "Earth",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
        ]
        for i in range(len(ss)):
            pos = state_vec[i * 3 : i * 3 + 3]
            vel = state_vec[(len(ss) + i) * 3 : (len(ss) + i) * 3 + 3]
            isv = np.append(pos, vel)
            isv = np.append(isv, masses[i])
            data_ic[ss[i]] = isv.tolist()

        data["physical_params"] = data_phy
        data["integration"] = data_int
        data["initial_conds"] = data_ic

        toml.dump(data, open(filename, "w"))
