#include "common.h"
#include "integrator_wisdom_holman.h"

/*
####################################################################################################
# Propagate Keplerian states using f and g functions:
#
# INPUT:
#  - psi: universal variable
#
# OUTPUT:
#  - c2, c3: auxiliary C2 and C3 functions
#
*/
void compute_c2c3(real psi, real *cs) {
    real c2, c3;
    if (psi > 1.e-10){
        c2 = (1.0 - cos(sqrt(psi))) / psi;
        c3 = (sqrt(psi) - sin(sqrt(psi))) / sqrt(pow(psi, 3.0));

    } else {
        if (psi < -1.e-6){
            c2 = (1 - cosh(sqrt(-psi))) / psi;
            c3 = (sinh(sqrt(-psi)) - sqrt(-psi)) / sqrt(-pow(psi, 3.0));
        } else {
           c2 = 0.5;
           c3 = 1.0 / 6.0;
        }
    }
    cs[0] = c2; cs[1] = c3;
    return;
}

/*
####################################################################################################
# Propagate Keplerian states using f and g functions:
#
# INPUT:
#  - t0: initial time
#  - tf: final time
#  - vr0: initial position vector
#  - vv0: initial velocity vector
#  - gm: gravitational parameter, G * (M + m)
#
# OUTPUT:
#  - vrf: final position vector
#  - vvf: final velocity vector
#
*/
// void propagate_kepler(real t0, real tf, real *vr0, real *vv0, real gm, real *vrf, real *vvf, size_t N) {
void propagate_kepler(real *jacobi_pos, real *jacobi_vel, real gm, real dt, size_t N, size_t particle_id) {

    real vr0[3];
    real vv0[3];
    for (size_t i = 0; i < 3; i++) vr0[i] = jacobi_pos[3 * particle_id + i];
    for (size_t i = 0; i < 3; i++) vv0[i] = jacobi_vel[3 * particle_id + i];

    // Compute the magnitude of the initial position and velocity vectors:
    real r0 = vector_norm(vr0, 3);
    real v0 = vector_norm(vv0, 3);

    // Precompute sqrt(gm):
    real sqrtgm = sqrt(gm);

    // Initial value of the Keplerian energy:
    real xi = 0.5 * v0 * v0  - gm / r0;

    // Semimajor axis:
    real sma = -gm / (2 * xi);
    real alpha = 1.0 / sma;
    real chi0;
    if (alpha > tol_energy){
        // Elliptic orbits:
        chi0 = sqrtgm * dt * alpha;
    } else if (alpha < tol_energy) {
        // Hyperbolic orbits:
        chi0 = sign(dt) * (sqrt(-sma) * log(-2.0 * gm * alpha * dt / ( dot(vr0, vv0) + sqrt(-gm * sma) * (1.0 - r0 * alpha))));
    } else {
        // Parabolic orbits:
        real cn = cross_norm(vr0, vv0);
        real p = cn * cn / gm;
        real s = 0.5 * atan( 1.0 / (3.0 * sqrt(gm / pow(p, 3.0)) * dt));
        real w = atan(pow(tan(s), 1.0 / 3.0));
        chi0 = sqrt(p) * 2 / tan(2 * w);
    }

    // Solve Kepler's equation:
    real cs[2];
    real chi = 0.0, r = 0.0, c2 = 0.0, c3 = 0.0, psi = 0.0;
    for (size_t j = 0; j < MAX_KEPLER_ITERATION; j++) {
        // Compute universal variable:
        psi = chi0 * chi0 * alpha;

        // Compute C2 and C3:
        compute_c2c3(psi, cs); c2 = cs[0]; c3 = cs[1];

        // Propagate radial distance:
        r = chi0 * chi0 * c2 + dot(vr0, vv0) / sqrtgm * chi0 * (1.0 - psi * c3) + r0 * (1 - psi * c2);

        // Auxiliary variable for f and g functions:
        chi = chi0 + (sqrtgm * dt - pow(chi0, 3.0) * c3 - dot(vr0, vv0) / sqrtgm * pow(chi0, 2.0) * c2 - r0 * chi0 * (1.0 - psi * c3)) / r;

        // Convergence:
        if (fabs(chi - chi0) < tol){
            break;
        }

        chi0 = chi;
    }

    if (fabs(chi - chi0) > tol) {
#ifdef LONGDOUBLE
        printf("WARNING: failed to solver Kepler's equation, error = %23.15Lg\n", fabs(chi - chi0));
#else
        printf("WARNING: failed to solver Kepler's equation, error = %23.15g\n", fabs(chi - chi0));
#endif
    }

    // Compute f and g functions, together with their derivatives:
    real f  = 1.0 - chi * chi / r0 * c2;
    real g  = dt - pow(chi, 3.0) / sqrtgm * c3;
    real dg = 1.0 - chi * chi / r * c2;
    real df = sqrtgm / (r * r0) * chi * (psi * c3 - 1.0);

    // Propagate states:
    for (size_t i = 0; i < 3; i++) {
        jacobi_pos[3 * particle_id + i] = f * vr0[i] + g * vv0[i];
        jacobi_vel[3 * particle_id + i] = df * vr0[i] + dg * vv0[i];
    }

    return;
}



/*
####################################################################################################
# Apply momentum kick following the Wisdom-Holman mapping strategy.
#
# INPUT:
#  - x: current state (heliocentric coordinates)
#  - dt: time step (local)
#  - masses: masses of the bodies
#  - nbodies: number of bodies
#  - accel: acceleration from H_interaction
#
# OUTPUT:
#  - kick: state at t + dt after the kick
#
*/
void wh_kick(real *vel, real dt, real *masses, size_t nbodies, real *accel) {
    // Change the momenta:
    for (size_t i = 3; i < 3 * nbodies; i++) {
        vel[i] += accel[i] * dt;
    }
    return;
}

/*
####################################################################################################
# Drift, i.e. Keplerian propagation.
#
# INPUT:
#  - x: current state (heliocentric coordinates)
#  - dt: time step (local)
#  - masses: masses of the bodies
#  - nbodies: number of bodies
#  - G: gravitational constant
#
# OUTPUT:
#  - drift: state at t + dt after drift
#
*/
void wh_drift(real *jacobi_pos, real *jacobi_vel, real dt, real *masses, size_t nbodies, real G) {

    // Propagate each body assuming Keplerina motion:
    real eta0 = masses[0];
    for (size_t i = 1; i < nbodies; i++) {
        // Interior mass:
        real eta = eta0 + masses[i];

        // Compute equivalent GM:
        real gm = G * masses[0] * eta / eta0;

        // Propagate:
        propagate_kepler(jacobi_pos, jacobi_vel, gm, dt, nbodies, i);

        eta0 = eta;
    }
    return;
}

/*
####################################################################################################
# Transform from heliocentric to Jacobi coordinates.
#
# INPUT:
#  - x: state in heliocentric coordinates
#  - masses: masses of the bodies
#  - nbodies: number of bodies
#
# OUTPUT:
#  - jacobi: state in Jacobi coordinates
#
*/
void helio2jacobi(real *pos, real *vel, real *masses, size_t nbodies, real *jacobi_pos, real *jacobi_vel) {


    // Compute eta (interior masses):
    real eta[nbodies];
    eta[0] = masses[0];
    for (size_t i = 1; i < nbodies; i++) {
        eta[i] = masses[i] + eta[i - 1];
    }

    // Assume central body at rest:
    jacobi_pos[0] = 0.0; jacobi_pos[1] = 0.0; jacobi_pos[2] = 0.0;
    jacobi_vel[0] = 0.0; jacobi_vel[1] = 0.0; jacobi_vel[2] = 0.0;

    // the jacobi coordinates for the second body is the same as the heliocentric
    jacobi_pos[3] = pos[3]; jacobi_pos[4] = pos[4]; jacobi_pos[5] = pos[5];
    jacobi_vel[3] = vel[3]; jacobi_vel[4] = vel[4]; jacobi_vel[5] = vel[5];

    // Jacobi coordinates of first body coincide with heliocentric, leave as they are.

    // Compute internal c.o.m. and momentum:
    real auxR[3], auxV[3], Ri[3], Vi[3];
    auxR[0] = masses[1] * pos[3];
    auxR[1] = masses[1] * pos[4];
    auxR[2] = masses[1] * pos[5];
    auxV[0] = masses[1] * vel[3];
    auxV[1] = masses[1] * vel[4];
    auxV[2] = masses[1] * vel[5];
    Ri[0] = auxR[0] / eta[1];
    Ri[1] = auxR[1] / eta[1];
    Ri[2] = auxR[2] / eta[1];
    Vi[0] = auxV[0] / eta[1];
    Vi[1] = auxV[1] / eta[1];
    Vi[2] = auxV[2] / eta[1];
    for (size_t i = 2; i < nbodies; i++) {
        jacobi_pos[3 * i] = pos[3 * i] - Ri[0];
        jacobi_pos[3 * i + 1] = pos[3 * i + 1] - Ri[1];
        jacobi_pos[3 * i + 2] = pos[3 * i + 2] - Ri[2];
        jacobi_vel[3 * i] = vel[3 * i] - Vi[0];
        jacobi_vel[3 * i + 1] = vel[3 * i + 1] - Vi[1];
        jacobi_vel[3 * i + 2] = vel[3 * i + 2] - Vi[2];

        // Compute the next internal c.o.m. and momentum of the sequence:
        if (i < nbodies - 1) {
            auxR[0] += (masses[i] * pos[3 * i]);
            auxR[1] += (masses[i] * pos[3 * i + 1]);
            auxR[2] += (masses[i] * pos[3 * i + 2]);
            auxV[0] += (masses[i] * vel[3 * i]);
            auxV[1] += (masses[i] * vel[3 * i + 1]);
            auxV[2] += (masses[i] * vel[3 * i + 2]);
            Ri[0] = auxR[0] / eta[i];
            Ri[1] = auxR[1] / eta[i];
            Ri[2] = auxR[2] / eta[i];
            Vi[0] = auxV[0] / eta[i];
            Vi[1] = auxV[1] / eta[i];
            Vi[2] = auxV[2] / eta[i];
        }
    }
    return;
}

/*
####################################################################################################
# Transform from Jacobi to heliocentric coordinates.
#
# INPUT:
#  - x: state in Jacobi coordinates
#  - masses: masses of the bodies
#  - nbodies: number of bodies
#
# OUTPUT:
#  - helio: state in heliocentric coordinates
#
*/
void jacobi2helio(real *jacobi_pos, real *jacobi_vel, real *masses, size_t nbodies, real *pos, real *vel) {

    // the helio coordinates for the second body is the same as the jacobi
    pos[3] = jacobi_pos[3]; pos[4] = jacobi_pos[4]; pos[5] = jacobi_pos[5];
    vel[3] = jacobi_vel[3]; vel[4] = jacobi_vel[4]; vel[5] = jacobi_vel[5];

    // Compute etas (interior masses):
    real eta[nbodies];
    eta[0] = masses[0];
    for (size_t i = 1; i < nbodies; i++) {
        eta[i] = masses[i] + eta[i - 1];
    }

    // Assume central body at rest:
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 0.0;
    vel[0] = 0.0;
    vel[1] = 0.0;
    vel[2] = 0.0;

    // Heliocentric coordinates of first body coincide with Jacobi, leave as they are.

    // Compute internal c.o.m. and momentum:
    real Ri[3], Vi[3];
    Ri[0] = masses[1] * jacobi_pos[3] / eta[1];
    Ri[1] = masses[1] * jacobi_pos[4] / eta[1];
    Ri[2] = masses[1] * jacobi_pos[5] / eta[1];
    Vi[0] = masses[1] * jacobi_vel[3] / eta[1];
    Vi[1] = masses[1] * jacobi_vel[4] / eta[1];
    Vi[2] = masses[1] * jacobi_vel[5] / eta[1];
    for (size_t i = 2; i < nbodies; i++) {
        pos[3 * i] = jacobi_pos[3 * i] + Ri[0];
        pos[3 * i + 1] = jacobi_pos[3 * i + 1] + Ri[1];
        pos[3 * i + 2] = jacobi_pos[3 * i + 2] + Ri[2];
        vel[3 * i] = jacobi_vel[3 * i] + Vi[0];
        vel[3 * i + 1] = jacobi_vel[3 * i + 1] + Vi[1];
        vel[3 * i + 2] = jacobi_vel[3 * i + 2] + Vi[2];

        // Compute the next internal c.o.m. and momentum of the sequence:
        if (i < nbodies - 1) {
            Ri[0] += masses[i] * jacobi_pos[3 * i] / eta[i];
            Ri[1] += masses[i] * jacobi_pos[3 * i + 1] / eta[i];
            Ri[2] += masses[i] * jacobi_pos[3 * i + 2] / eta[i];
            Vi[0] += masses[i] * jacobi_vel[3 * i] / eta[i];
            Vi[1] += masses[i] * jacobi_vel[3 * i + 1] / eta[i];
            Vi[2] += masses[i] * jacobi_vel[3 * i + 2] / eta[i];
        }
        for (size_t i = 0; i < 3; i++) {
        }
    }
    return;
}

/*
####################################################################################################
# Compute acceleration on all bodies.
#
# INPUT:
#  - helio: current state in heliocentric coordinates
#  - jac: current state in Jacobi coordinates
#  - masses: masses of the bodies
#  - nbodies: number of bodies
#  - G: gravitational constant
#
# OUTPUT:
#  - accel: acceleration vector
#
*/
void compute_accel(real *pos, real *jacobi_pos, real *masses, size_t nbodies, real G, real *accel) {

    // Allocate:
    // accel = np.zeros(nbodies * 3)

    // Acceleration of first body is assumed zero:
    real inv_r3helio[nbodies];
    real inv_r3jac[nbodies];
    // real inv_rhelio[nbodies];
    // real inv_rjac[nbodies];
    real *inv_rhelio = inv_r3helio;
    real *inv_rjac = inv_r3jac;
    for (size_t i = 0; i < nbodies; i++) {
        if (i < 2) {
            inv_rhelio[i] = 0.0;
            inv_r3helio[i] = 0.0;
            inv_rjac[i] = 0.0;
            inv_r3jac[i] = 0.0;
        } else {
            inv_rhelio[i] = 1.0 / vector_norm(&(pos[3 * i]), 3);
            inv_r3helio[i] = pow(inv_rhelio[i], 3.0);
            inv_rjac[i] = 1.0 / vector_norm(&(jacobi_pos[3 * i]), 3);
            inv_r3jac[i] = pow(inv_rjac[i], 3.0);
        }
    }

    // Compute all indirect terms at once:
    real accel_ind[3 * nbodies];
    real accel_ind_tmp[3];
    accel_ind_tmp[0] = 0.0; accel_ind_tmp[1] = 0.0; accel_ind_tmp[2] = 0.0;
    for (size_t i = 0; i < 3 * nbodies; i++) accel_ind[i] = 0.0;
    for (size_t i = 2; i < nbodies; i++) {
        accel_ind_tmp[0] -= G * masses[i] * pos[3 * i] * inv_r3helio[i];
        accel_ind_tmp[1] -= G * masses[i] * pos[3 * i + 1] * inv_r3helio[i];
        accel_ind_tmp[2] -= G * masses[i] * pos[3 * i + 2] * inv_r3helio[i];
    }
    for (size_t i = 3; i < 3 * nbodies; i++) {
        accel_ind[i] = accel_ind_tmp[i % 3];
    }


    // Compute contribution from central body:
    real accel_cent[3 * nbodies];
    for (size_t i = 0; i < 3 * nbodies; i++) accel_cent[i] = 0.0;
    for (size_t i = 2; i < nbodies; i++) {
        accel_cent[3 * i] = G * masses[0] * (jacobi_pos[3 * i] * inv_r3jac[i] - pos[3 * i] * inv_r3helio[i]);
        accel_cent[3 * i + 1] = G * masses[0] * (jacobi_pos[3 * i + 1] * inv_r3jac[i] - pos[3 * i + 1] * inv_r3helio[i]);
        accel_cent[3 * i + 2] = G * masses[0] * (jacobi_pos[3 * i + 2] * inv_r3jac[i] - pos[3 * i + 2] * inv_r3helio[i]);
    }

    // Compute third part of the Hamiltonian:
    real accel2[3 * nbodies];
    for (size_t i = 0; i < 3 * nbodies; i++) accel2[i] = 0.0;
    real etai = masses[0];
    for (size_t i = 2; i < nbodies; i++) {
        etai += masses[i - 1];
        accel2[3 * i] = accel2[3 * (i - 1)] + G * masses[i] * masses[0] * inv_r3jac[i] / etai * jacobi_pos[3 * i];
        accel2[3 * i + 1] = accel2[3 * (i - 1) + 1] + G * masses[i] * masses[0] * inv_r3jac[i] / etai * jacobi_pos[3 * i + 1];
        accel2[3 * i + 2] = accel2[3 * (i - 1) + 2] + G * masses[i] * masses[0] * inv_r3jac[i] / etai * jacobi_pos[3 * i + 2];
    }

    // Compute final part of the Hamiltonian:
    real accel3[3 * nbodies];
    real diff[3];
    real aux;
    for (size_t i = 0; i < 3 * nbodies; i++) accel3[i] = 0.0;
    for (size_t i = 1; i < nbodies - 1; i++) {
        for (size_t j = i + 1; j < nbodies; j++) {
            diff[0] = pos[3 * j] - pos[3 * i];
            diff[1] = pos[3 * j + 1] - pos[3 * i + 1];
            diff[2] = pos[3 * j + 2] - pos[3 * i + 2];
            aux = 1.0 / pow(vector_norm(diff, 3), 3.0);
            accel3[3 * j] -= G * masses[i] * aux * diff[0];
            accel3[3 * j + 1] -= G * masses[i] * aux * diff[1];
            accel3[3 * j + 2] -= G * masses[i] * aux * diff[2];
            accel3[3 * i] += G * masses[j] * aux * diff[0];
            accel3[3 * i + 1] += G * masses[j] * aux * diff[1];
            accel3[3 * i + 2] += G * masses[j] * aux * diff[2];
        }
    }

    // Add all contributions:
    for (size_t i = 0; i < 3 * nbodies; i++) {
        accel[i] = accel_ind[i] + accel_cent[i] + accel2[i] + accel3[i];
    }

    return;
}

/*
####################################################################################################
# Advance one step using the Wisdom-Holman mapping. Implements a Kick-Drift-Kick strategy.
#
# INPUT:
#  - x: current state (heliocentric coordinates)
#  - t: current time
#  - dt: time step
#  - masses: masses of the bodies
#  - nbodies: number of bodies
#  - accel: acceleration from H_interaction
#  - G: gravitational constant
#
# OUTPUT:
#  - helio: heliocentric state at t + dt
#  - accel: updated acceleration at dt
#
*/
void wh_advance_step(real *pos, real *vel, real *jacobi_pos, real *jacobi_vel, real t, real dt, real *masses, size_t nbodies, real *accel, real G){

    // Kick:
    wh_kick(vel, dt / 2, masses, nbodies, accel);

    // Convert from heliocentric to Jacobi for drifting:
    helio2jacobi(pos, vel, masses, nbodies, jacobi_pos, jacobi_vel);

    // Drift
    wh_drift(jacobi_pos, jacobi_vel, dt, masses, nbodies, G);

    // Convert from Jacobi to heliocentric for kicking:
    jacobi2helio(jacobi_pos, jacobi_vel, masses, nbodies, pos, vel);

    // Compute acceleration at t + dt:
    compute_accel(pos, jacobi_pos, masses, nbodies, G, accel);

    // Kick:
    wh_kick(vel, dt / 2, masses, nbodies, accel);

    return;
}

/*
####################################################################################################
# Propagate orbit using the WH symplectic mapping.
#
# INPUT:
#  - x: current state (heliocentric coordinates)
#  - t: current time
#  - tf: final time
#  - dt: time step
#  - masses: masses of the bodies
#  - nbodies: number of bodies
#  - skip_step: integer factor, store every 'skip_step' points instead of all of them
#  - G: gravitational constant
#
# OUTPUT:
#  - sol_time: time steps in which the solution is provided
#  - sol_state: solution, state vectors at sol_time
#
*/
void integrator_wisdom_holman(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real _G, real _t, real _t_end, real _dt) {
    // allocation
    real jacobi_pos[3 * N];
    real jacobi_vel[3 * N];
    real accel[3 * N];

    // Initialize: compute Jacobi coordinates and initial acceleration:
    for (size_t i = 0; i < 3 * N; i++) accel[i] = 0.0;
    for (size_t i = 0; i < 3 * N; i++) jacobi_pos[i] = 0.0;
    for (size_t i = 0; i < 3 * N; i++) jacobi_vel[i] = 0.0;

    helio2jacobi(pos, vel, m_vec, N, jacobi_pos, jacobi_vel);
    compute_accel(pos, jacobi_pos, m_vec, N, _G, accel);

    // Main loop:
    while (_t < _t_end) {
        // Advance one step:
        wh_advance_step(pos, vel, jacobi_pos, jacobi_vel, _t, _dt, m_vec, N, accel, _G);

        // Advance time:
        _t += _dt;
        t_global = _t;
    }
    return;
}
