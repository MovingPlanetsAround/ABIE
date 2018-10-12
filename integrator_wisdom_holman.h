#ifndef INTEGRATOR_WISDOM_HOLMAN
#define INTEGRATOR_WISDOM_HOLMAN

double h;

// Internal tolerance for solving Kepler's equation:
real tol = 1e-12;

// Energy tolerance: used to distinguish between elliptic, parabolic, and hyperbolic orbits,
// ideally 0:
real tol_energy = 0.0;
size_t MAX_KEPLER_ITERATION = 500;
void integrator_wisdom_holman(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real _G, real _t, real _t_end, real _dt);

#endif
