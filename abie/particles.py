import numpy as np
from abie.particle import Particle
from abie.tools import Tools
from six import string_types


class Particles(object):
    """
    A particle container.
    """

    def __init__(self, const_g):
        self.__particles = []
        self.__positions = np.array([])
        self.__velocities = np.array([])
        self.__masses = np.array([])
        self.__names = dict()
        self.__N = 0
        self.CONST_G = const_g
        self.primary = "#COM#"  # '#COM#', '#M_MAX#', '#M_MIN#', or name/ID

    def __repr__(self):
        str_concat = "Total number of particles: %d\n" % len(self.__particles)
        for pid, p in enumerate(self.__particles):
            str_concat += p.__repr__() + "\n"
        return str_concat

    @property
    def N(self):
        return self.__N

    @property
    def particles(self):
        return self

    @property
    def positions(self):
        return self.__positions

    @property
    def names(self):
        name_list = []
        for p in self.__particles:
            name_list.append(p.name)
        return name_list

    @property
    def hashes(self):
        hash_list = []
        for p in self.__particles:
            hash_list.append(p.hash)
        return hash_list

    @property
    def radii(self):
        radii_list = np.empty(self.__N)
        for p_id, p in enumerate(self.__particles):
            radii_list[p_id] = p.radius
        return radii_list

    @property
    def ptypes(self):
        ptype_list = []
        for p in self.__particles:
            ptype_list.append(p.ptype)
        return ptype_list

    @positions.setter
    def positions(self, pos_vec):
        """
        Set the positions of all particles with a flattened position vector.
        :param pos_vec: The position vector
        :return:
        """
        if type(pos_vec).__module__ == np.__name__:
            if pos_vec.size == 3 * self.__N:
                for p_id in range(self.__N):
                    self.__particles[p_id].pos = pos_vec[3 * p_id : 3 * p_id + 3]
                self.__positions = pos_vec
            else:
                raise ValueError("Position vector must be len=3 vector.")
        else:
            raise TypeError("Position vector must be a numpy vector with len=3*N.")

    @property
    def velocities(self):
        return self.__velocities

    @velocities.setter
    def velocities(self, vel_vec):
        """
        Set the velocities of all particles with a flattened velocity vector.
        :param vel_vec: The velocity vector
        :return:
        """
        if type(vel_vec).__module__ == np.__name__:
            if vel_vec.size == 3 * self.__N:
                for p_id in range(self.__N):
                    self.__particles[p_id].vel = vel_vec[3 * p_id : 3 * p_id + 3]
                self.__velocities = vel_vec
            else:
                raise ValueError("Velocity vector must be len=3*N vector.")
        else:
            raise TypeError("Position vector must be a numpy vector with len=3.")

    @property
    def masses(self):
        return self.__masses

    def add(
        self,
        pos=np.zeros(3),
        vel=np.zeros(3),
        mass=0.0,
        name=None,
        radius=0.0,
        ptype=0,
        a=None,
        e=0.0,
        i=0.0,
        Omega=0.0,
        omega=0.0,
        f=0.0,
        primary=None,
    ):
        """
        High-level routine to add a particle to the global set according to the pos/vel or elements.
        :param pos: The 3D position vector of the particle
        :param vel: The 3D velocity vector of the particle
        :param mass: The mass of the particle
        :param name: The name of the particle. Could be either numeric or string
        :param radius: The radius of the particle
        :param ptype: The particle type; 0 = normal particle; 1 = test particle; 2 = low-mass particle
        :param a: Semi-major axis of the particle
        :param e: Eccentricity of the particle
        :param i: Inclination of the particle (radians)
        :param Omega: Longitude of the ascending node of the particle
        :param omega: Argument of periapsis of the particle
        :param f:
        :param primary: The primary object on which the orbital elements are defined
        :return:
        """
        primary_original = primary
        if a is not None:
            # determine the primary body
            primary = self.determine_primary_body(primary)

            # convert orbital elements to Cartesian coordinates
            pos, vel = Tools.from_orbital_elements_to_cartesian(
                mp=mass,
                ms=primary.mass,
                semimajor_axis=a,
                eccentricity=e,
                inclination=i,
                longitude_of_ascending_node=Omega,
                argument_of_periapsis=omega,
                true_anomaly=f,
                G=self.CONST_G,
            )
            pos += primary.pos
            vel += primary.vel
        # print name, pos, vel, mass

        self.__positions = np.append(self.__positions, pos)
        self.__velocities = np.append(self.__velocities, vel)
        self.__masses = np.append(self.__masses, np.array([mass]))
        if name is not None:
            self.__names[name] = self.__N
        particle = Particle(
            mass=mass,
            pos=self.__positions[3 * self.__N : 3 * self.__N + 3],
            vel=self.__velocities[3 * self.__N : 3 * self.__N + 3],
            name=name,
            radius=radius,
            ptype=ptype,
            primary=primary_original,
        )
        self.__particles.append(particle)
        self.__N += 1

    def add_particle(self, particle):
        """
        Low-level routine to add a particle to the global particle set
        :param particle: The particle to add
        :return:
        """
        if isinstance(particle, Particle) and (particle not in self.particles):
            self.__positions = np.append(self.__positions, particle.pos)
            self.__velocities = np.append(self.__velocities, particle.vel)
            self.__masses = np.append(self.__masses, np.array([particle.mass]))
            if particle.name is not None:
                self.__names[particle.name] = self.__N
            particle.pos = self.__positions[3 * self.__N, 3 * self.__N + 3]
            particle.vel = self.__velocities[3 * self.__N + 3, 3 * self.__N + 6]
            self.__particles.append(particle)
            self.__N += 1
        else:
            raise TypeError("Incompatible particle type.")

    def remove_particle(self, particle):
        """
        Low-level routine to remove a particle.
        :param particle: The particle to remove
        :return:
        """
        if isinstance(particle, Particle) and (particle in self.particles):
            pid = self.__particles.index(particle)
            self.__positions = np.delete(self.__positions, range(3 * pid, 3 * pid + 3))
            self.__velocities = np.delete(
                self.__velocities, range(3 * pid, 3 * pid + 3)
            )
            self.__masses = np.delete(self.__masses, pid)
            self.__particles.remove(particle)
            self.__N -= 1
        else:
            raise TypeError("Incompatible particle type.")

    def merge_particles_inelastically(self, pid1, pid2):
        """
        Merge particles with IDs (pid1, pid2) inelastically, conserving momentum but not energy.
        The less massive particle will be deleted. If some other particles refer to the removed particle as the
        center of mass, then the new center-of-mass will be set to the merged particle.

        :param pid1: The ID of the first particle
        :param pid2: The ID of the second particle
        :return:
        """
        # TODO: ensure that pid1 and pid2 refer to the same particles even if particles are removed
        # TODO: add a hash table for the particle set to quickly find particles uniquely
        try:
            p1 = self.particles[pid1]
            p2 = self.particles[pid2]
        except ValueError:
            return -1
        if p1 is not None and p2 is not None:
            if p1.mass >= p2.mass:
                # merge into p1
                p1.vel = (p1.mass * p1.vel + p2.mass * p2.vel) / (p1.mass + p2.mass)
                self.velocities[3 * pid1 : 3 * pid1 + 3] = p1.vel
                p1.mass = p1.mass + p2.mass
                self.masses[pid1] = p1.mass
                p1.radius = np.power(
                    np.power(p1.radius, 3.0) + np.power(p2.radius, 3.0), 1.0 / 3
                )
                self.radii[pid1] = p1.radius
                # search for objects that defines their orbital elements with respect to p2, and change to p1
                for p_id, p in enumerate(self.__particles):
                    if (p.primary is not None) and (
                        self.particles[p.primary] == pid2
                        or self.particles[p.primary] == p2.name
                    ):
                        p.primary = pid1
                self.remove_particle(p2)
                print(
                    (
                        "Merging particles inelastically: #%d + #%d ==> #%d"
                        % (pid1, pid2, pid1)
                    )
                )
            else:
                # merge into p2
                p2.vel = (p1.mass * p1.vel + p2.mass * p2.vel) / (p1.mass + p2.mass)
                self.velocities[3 * pid2 : 3 * pid2 + 3] = p2.vel
                p2.mass = p1.mass + p2.mass
                self.masses[pid2] = p2.mass
                p2.radius = np.power(
                    np.power(p1.radius, 3.0) + np.power(p2.radius, 3.0), 1.0 / 3
                )
                self.radii[pid2] = p2.radius
                # search for objects that defines their orbital elements with respect to p2, and change to p1
                for p in self.__particles:
                    if (p.primary is not None) and (
                        self.particles[p.primary] == pid1
                        or self.particles[p.primary] == p1.name
                    ):
                        p.primary = pid2
                self.remove_particle(p1)
                print(
                    (
                        "Merging particles inelastically: #%d + #%d ==> #%d"
                        % (pid2, pid1, pid2)
                    )
                )
            return 0
        else:
            return -1

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < len(self.__particles):
                return self.__particles[item]
            else:
                raise ValueError("Particle #%d does not exist!" % item)
        elif isinstance(item, string_types):
            if item in self.__names:
                return self.__particles[self.__names[item]]
            else:
                raise ValueError("Particle %s not exist!" % item)
        return None

    def get_center_of_mass(self, subset=None):
        """
        Compute the center-of-mass. If subset is not given, compute the
        center-of-mass of the entire system. Otherwise, compute the COM of
        the subset. Subset is a list of particle IDs in the system.
        """
        com_pos = np.array([0.0, 0.0, 0.0])
        com_vel = np.array([0.0, 0.0, 0.0])
        if subset is None:
            for pid in range(0, self.N):
                com_pos += self.particles[pid].pos * self.particles[pid].mass
                com_vel += self.particles[pid].vel * self.particles[pid].mass
            com_pos /= np.sum(self.masses)
            com_vel /= np.sum(self.masses)
            return Particle(mass=np.sum(self.masses), pos=com_pos, vel=com_vel)
        else:
            subset_masses = 0.0
            for pid in subset:
                com_pos += self.particles[pid].pos * self.particles[pid].mass
                com_vel += self.particles[pid].vel * self.particles[pid].mass
                subset_masses += self.particles[pid].mass
            com_pos /= subset_masses
            com_vel /= subset_masses
            return Particle(mass=subset_masses, pos=com_pos, vel=com_vel)

    def calculate_orbital_elements(self, primary=None):
        # calculate the orbital elements
        orbital_elem = np.zeros((self.N, 6))
        if self.N < 2:
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        for pid in range(0, self.N):
            p = self.particles[pid]
            if p.primary is not None:
                # if the particle itself has its own primary object, use its own primary object
                primary = self.determine_primary_body(p.primary)
            else:
                # else use the globally-defined primary
                primary = self.determine_primary_body(primary)
            a, e, f, i, om, Om = Tools.from_cartesian_to_orbital_elements(
                mp=p.mass,
                ms=primary.mass,
                position=np.array([p.x - primary.x, p.y - primary.y, p.z - primary.z]),
                velocity=np.array(
                    [p.vx - primary.vx, p.vy - primary.vy, p.vz - primary.vz]
                ),
                G=self.CONST_G,
            )
            orbital_elem[pid, :] = np.array([a, e, i, Om, om, f])
        return orbital_elem

    def calculate_aei(self, primary=None):
        # calculate the orbital elements
        orbital_elem = np.zeros((self.N, 3))
        if self.N < 2:
            return np.array([np.nan, np.nan, np.nan])

        for pid in range(0, self.N):
            p = self.particles[pid]
            if p.primary is not None:
                # if the particle itself has its own primary object, use its own primary object
                primary = self.determine_primary_body(p.primary)
            else:
                # else use the globally-defined primary
                primary = self.determine_primary_body(primary)
            a, e, i = Tools.from_cartesian_to_aei(
                mp=p.mass,
                ms=primary.mass,
                position=np.array([p.x - primary.x, p.y - primary.y, p.z - primary.z]),
                velocity=np.array(
                    [p.vx - primary.vx, p.vy - primary.vy, p.vz - primary.vz]
                ),
                G=self.CONST_G,
            )
            orbital_elem[pid, :] = np.array([a, e, i])
        return orbital_elem

    def determine_primary_body(self, primary):
        if primary is None:
            if self.primary == "#COM#":
                primary = self.get_center_of_mass()
            elif self.primary == "#M_MAX#":
                primary = self.particles[np.argmax(self.masses)]
            elif self.primary == "#M_MIN#":
                primary = self.particles[np.argmin(self.masses)]
            elif isinstance(self.primary, list):
                primary = self.get_center_of_mass(subset=self.primary)
            else:
                primary = self.__getitem__(self.primary)
        elif isinstance(primary, string_types):
            if self.__getitem__(primary) is None:
                raise ValueError("Object with the name %s not found!" % primary)
            else:
                primary = self.__getitem__(primary)
        elif isinstance(primary, list):
            primary = self.get_center_of_mass(subset=primary)
        return primary

    @property
    def energy(self):
        energy = 0.0
        for i in range(0, self.N):
            e_kin = 0.5 * self.masses[i] * np.linalg.norm(self.particles[i].vel) ** 2
            energy += e_kin
            for j in range(0, self.N):
                if i == j:
                    continue
                e_pot = (
                    0.5
                    * self.CONST_G
                    * self.masses[i]
                    * self.masses[j]
                    / np.linalg.norm(self.particles[i].pos - self.particles[j].pos)
                )
                energy -= e_pot
        return energy
