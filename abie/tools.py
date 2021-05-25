import numpy as np


class Tools(object):
    @staticmethod
    def compute_energy(sol_state, masses, const_g):
        n_rows, n_cols = sol_state.shape
        if n_cols % 6 == 0 and n_rows > 0:
            energy = np.zeros(n_rows)
            for t in range(n_rows):
                for i in range(0, n_cols / 6):
                    energy[t] += (
                        0.5
                        * masses[i]
                        * np.linalg.norm(
                            sol_state[t][n_cols / 2 + i * 3 : n_cols / 2 + i * 3 + 3]
                        )
                        ** 2
                    )
                    for j in range(0, n_cols / 6):
                        if i == j:
                            continue
                        energy[t] -= (
                            0.5
                            * const_g
                            * masses[i]
                            * masses[j]
                            / np.linalg.norm(
                                sol_state[t][i * 3 : i * 3 + 3]
                                - sol_state[t][j * 3 : j * 3 + 3]
                            )
                        )
            return energy

    @staticmethod
    def from_orbital_elements_to_cartesian(
        mp,
        ms,
        semimajor_axis,
        eccentricity,
        true_anomaly,
        inclination,
        argument_of_periapsis,
        longitude_of_ascending_node,
        G=4.0 * np.pi ** 2,
    ):
        """

        Function that returns position and velocities computed from the input orbital
        elements. Angles in radians, inclination between 0 and 180

        """

        cos_true_anomaly = np.cos(true_anomaly)
        sin_true_anomaly = np.sin(true_anomaly)

        cos_inclination = np.cos(inclination)
        sin_inclination = np.sin(inclination)

        cos_arg_per = np.cos(argument_of_periapsis)
        sin_arg_per = np.sin(argument_of_periapsis)

        cos_long_asc_nodes = np.cos(longitude_of_ascending_node)
        sin_long_asc_nodes = np.sin(longitude_of_ascending_node)

        ### e_vec is a unit vector directed towards periapsis ###
        e_vec_x = (
            cos_long_asc_nodes * cos_arg_per
            - sin_long_asc_nodes * sin_arg_per * cos_inclination
        )
        e_vec_y = (
            sin_long_asc_nodes * cos_arg_per
            + cos_long_asc_nodes * sin_arg_per * cos_inclination
        )
        e_vec_z = sin_arg_per * sin_inclination
        e_vec = np.array([e_vec_x, e_vec_y, e_vec_z])

        ### q is a unit vector perpendicular to e_vec and the orbital angular momentum vector ###
        q_vec_x = (
            -cos_long_asc_nodes * sin_arg_per
            - sin_long_asc_nodes * cos_arg_per * cos_inclination
        )
        q_vec_y = (
            -sin_long_asc_nodes * sin_arg_per
            + cos_long_asc_nodes * cos_arg_per * cos_inclination
        )
        q_vec_z = cos_arg_per * sin_inclination
        q_vec = np.array([q_vec_x, q_vec_y, q_vec_z])

        #    print 'alpha',alphax**2+alphay**2+alphaz**2 # For debugging; should be 1
        #    print 'beta',betax**2+betay**2+betaz**2 # For debugging; should be 1

        ### Relative position and velocity ###
        separation = (
            semimajor_axis
            * (1.0 - eccentricity ** 2)
            / (1.0 + eccentricity * cos_true_anomaly)
        )  # Compute the relative separation
        position_vector = (
            separation * cos_true_anomaly * e_vec
            + separation * sin_true_anomaly * q_vec
        )
        velocity_tilde = np.sqrt(
            G * (mp + ms) / (semimajor_axis * (1.0 - eccentricity ** 2))
        )  # Common factor
        velocity_vector = (
            -1.0 * velocity_tilde * sin_true_anomaly * e_vec
            + velocity_tilde * (eccentricity + cos_true_anomaly) * q_vec
        )

        return position_vector, velocity_vector

    @staticmethod
    def from_cartesian_to_orbital_elements(
        mp, ms, position, velocity, G=4.0 * np.pi ** 2
    ):
        """

        Function that computes orbital elements from cartesian coordinates.
        Return values are: mass1, mass2, semimajor axis, eccentricity,
        true anomaly, inclination, longitude of the ascending nodes and the
        argument of pericenter. All angles are returned in radians.
        In case of a perfectly circular orbit the true anomaly
        and argument of pericenter cannot be determined; in this case, the
        return values are 0.0 for both angles.

        """

        total_mass = mp + ms

        ### specific energy ###
        v_sq = np.dot(velocity, velocity)
        r_sq = np.dot(position, position)
        r = np.sqrt(r_sq)

        specific_energy = (1.0 / 2.0) * v_sq - G * total_mass / r
        # if specific_energy >= 0.0:
        #     print 'Not a bound orbit!'

        semimajor_axis = -G * total_mass / (2.0 * specific_energy)

        ### specific angular momentum ###
        specific_angular_momentum = np.cross(position, velocity)
        specific_angular_momentum_norm = np.sqrt(
            np.dot(specific_angular_momentum, specific_angular_momentum)
        )
        specific_angular_momentum_unit = (
            specific_angular_momentum / specific_angular_momentum_norm
        )

        maximum_specific_angular_momentum_norm = (
            G * total_mass / (np.sqrt(-2.0 * specific_energy))
        )
        ell = (
            specific_angular_momentum_norm / maximum_specific_angular_momentum_norm
        )  ### specific AM in units of maximum AM

        ### for e = 0 or e nearly 0, ell can be slightly larger than unity due to numerical reasons ###
        ell_epsilon = 1e-15

        completely_or_nearly_circular = False

        if ell > 1.0:
            if (
                1.0 < ell <= ell + ell_epsilon
            ):  ### still unity within numerical precision
                print(
                    "orbit is completely or nearly circular; in this case the LRL vector cannot be used to reliably obtain the argument of pericenter and true anomaly; the output values of the latter will be set to zero; output e will be e = 0"
                )
                ell = 1.0
                completely_or_nearly_circular = True
            else:  ### larger than unity within numerical precision
                raise Exception(
                    "angular momentum larger than maximum angular momentum for bound orbit"
                )

        eccentricity = np.sqrt(1.0 - ell ** 2)

        ### Orbital inclination ###
        z_vector = np.array([0.0, 0.0, 1.0])
        inclination = np.arccos(np.dot(z_vector, specific_angular_momentum_unit))

        ### Longitude of ascending nodes, with reference direction along x-axis ###
        ascending_node_vector = np.cross(z_vector, specific_angular_momentum)
        ascending_node_vector_norm = np.sqrt(
            np.dot(ascending_node_vector, ascending_node_vector)
        )
        if ascending_node_vector_norm == 0:
            ascending_node_vector_unit = np.array([1.0, 0.0, 0.0])
        else:
            ascending_node_vector_unit = (
                ascending_node_vector / ascending_node_vector_norm
            )

        long_asc_nodes = np.arctan2(
            ascending_node_vector_unit[1], ascending_node_vector_unit[0]
        )

        ### Argument of periapsis and true anomaly, using eccentricity a.k.a. Laplace-Runge-Lenz (LRL) vector ###
        mu = G * total_mass
        position_unit = position / r
        e_vector = (1.0 / mu) * np.cross(
            velocity, specific_angular_momentum
        ) - position_unit  ### Laplace-Runge-Lenz vector

        if (
            completely_or_nearly_circular == True
        ):  ### orbit is completely or nearly circular; in this case the LRL vector cannot be used to reliably obtain the argument of pericenter and true anomaly; the output values of the latter will be set to zero; output e will be e = 0
            arg_per = 0.0
            true_anomaly = 0.0
        else:
            e_vector_norm = np.sqrt(np.dot(e_vector, e_vector))
            e_vector_unit = e_vector / e_vector_norm

        e_vector_unit_cross_AM_unit = np.cross(
            e_vector_unit, specific_angular_momentum_unit
        )
        sin_arg_per = np.dot(ascending_node_vector_unit, e_vector_unit_cross_AM_unit)
        cos_arg_per = np.dot(e_vector_unit, ascending_node_vector_unit)
        arg_per = np.arctan2(sin_arg_per, cos_arg_per)

        sin_true_anomaly = np.dot(position_unit, -1.0 * e_vector_unit_cross_AM_unit)
        cos_true_anomaly = np.dot(position_unit, e_vector_unit)
        true_anomaly = np.arctan2(sin_true_anomaly, cos_true_anomaly)

        return (
            semimajor_axis,
            eccentricity,
            true_anomaly,
            inclination,
            arg_per,
            long_asc_nodes,
        )

    @staticmethod
    def from_cartesian_to_aei(
        mp,
        ms,
        position,
        velocity,
        G=0.0172020989 ** 2,
        z_vector=np.array([0.0, 0.0, 1.0]),
    ):
        """

        Function that computes orbital elements from cartesian coordinates.
        Return values are: mass1, mass2, semimajor axis, eccentricity,
        true anomaly, inclination, longitude of the ascending nodes and the
        argument of pericenter. All angles are returned in radians.
        In case of a perfectly circular orbit the true anomaly
        and argument of pericenter cannot be determined; in this case, the
        return values are 0.0 for both angles.

        """

        total_mass = mp + ms
        mu = G * total_mass

        if velocity.ndim == 1 and position.ndim == 1:
            # Calculate the a-e-i of one body wrt another one
            v_sq = np.dot(velocity, velocity)
            r_sq = np.dot(position, position)
            r = np.sqrt(r_sq)
            if r == 0:
                return np.nan, np.nan, np.nan

            specific_energy = (1.0 / 2.0) * v_sq - G * total_mass / r
            # if specific_energy >= 0.0:
            #     print 'Not a bound orbit!'

            semimajor_axis = -G * total_mass / (2.0 * specific_energy)

            specific_angular_momentum = np.cross(position, velocity)
            specific_angular_momentum_norm = np.sqrt(
                np.dot(specific_angular_momentum, specific_angular_momentum)
            )
            specific_angular_momentum_unit = (
                specific_angular_momentum / specific_angular_momentum_norm
            )
            ecc_vec = (
                (v_sq - mu / r) * position - np.dot(position, velocity) * velocity
            ) / mu

            inclination = np.arccos(np.dot(z_vector, specific_angular_momentum_unit))
            eccentricity = np.linalg.norm(ecc_vec)

        elif velocity.ndim == 2 and position.ndim == 2:

            # calculate a list of bodies (pos and vel are n*3 arrays, representing n points)
            v_sq = np.sum(velocity * velocity, axis=0)
            r_sq = np.sum(position * position, axis=0)
            r = np.sqrt(r_sq)

            specific_energy = (1.0 / 2.0) * v_sq - G * total_mass / r

            semimajor_axis = -G * total_mass / (2.0 * specific_energy)

            ### specific angular momentum ###
            specific_angular_momentum = np.cross(
                position.transpose(), velocity.transpose()
            ).transpose()
            specific_angular_momentum_norm = np.sqrt(
                np.sum(specific_angular_momentum * specific_angular_momentum, axis=0)
            )
            specific_angular_momentum_unit = (
                specific_angular_momentum / specific_angular_momentum_norm
            )
            ecc_vec = (
                (v_sq - mu / r) * position
                - np.sum(position * velocity, axis=0) * velocity
            ) / mu

            eccentricity = np.linalg.norm(ecc_vec, axis=0)

            ### Orbital inclination ###
            inclination = np.arccos(np.matmul(z_vector, specific_angular_momentum_unit))

        return semimajor_axis, eccentricity, inclination
