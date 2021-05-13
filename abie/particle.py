import numpy as np


class Particle(object):
    def __init__(
        self,
        ptype=0,
        mass=0.0,
        pos=np.zeros(3),
        vel=np.zeros(3),
        radius=0.0,
        name=None,
        primary=None,
    ):
        self.ptype = (
            ptype  # particle type. 0: regular particle; 1: massless; 2: low-mass
        )
        self.mass = mass  # mass
        self.radius = radius  # radius
        self.name = name  # user-assigned name, optional
        # self.hash = hash(self)  # unique key of the particle
        self.hash = np.random.randint(100000000, 999999999)
        self.primary = primary  # this defines the primary object that it is orbiting
        self.__pos = pos
        self.__vel = vel
        self.x = pos[0]  # x
        self.y = pos[1]  # y
        self.z = pos[2]  # z
        self.vx = vel[0]  # vx
        self.vy = vel[1]  # vy
        self.vz = vel[2]  # vz

    def __repr__(self):
        return "Particle(m={0:g}, x={1:g}, y={2:g}, z={3:g}, vx={4:g}, vy={5:g}, vz={6:g}, r={7:g}, name='{8:s}', hash={9:d})".format(
            self.mass,
            self.x,
            self.y,
            self.z,
            self.vx,
            self.vy,
            self.vz,
            self.radius,
            self.name,
            self.hash,
        )

    @property
    def pos(self):
        return self.__pos

    @property
    def vel(self):
        return self.__vel

    @pos.setter
    def pos(self, pos_vec):
        if type(pos_vec).__module__ == np.__name__:
            if pos_vec.size == 3:
                self.x = pos_vec[0]
                self.y = pos_vec[1]
                self.z = pos_vec[2]
                self.__pos = pos_vec
            else:
                raise ValueError("Position vector must be len=3 vector.")
        else:
            raise TypeError("Position vector must be a numpy vector with len=3.")

    @vel.setter
    def vel(self, vel_vec):
        if type(vel_vec).__module__ == np.__name__:
            if vel_vec.size == 3:
                self.vx = vel_vec[0]
                self.vy = vel_vec[1]
                self.vz = vel_vec[2]
                self.__vel = vel_vec
            else:
                raise ValueError("Velocity vector must be len=3 vector.")
        else:
            raise TypeError("Velocity vector must be a numpy vector with len=3.")
