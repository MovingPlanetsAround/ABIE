#ifndef INTEGRATOR_RUNGE_KUTTA
#define INTEGRATOR_RUNGE_KUTTA

// real *pos;  // the position state vector specified by the users
// real *vel;  // the velocity state vector specified by the users
// real *m_vec;

// size_t N; // number of particles
// size_t dim; // usually 3 * N
// size_t N_active; // number of massive particles
// real G; // gravitational constant
// real *eps; // per-particle softening parameter
// real t;
// real t_end;
double dt; // the time step specified by the user
// real close_encounter_distance = 0.0;  // if 0, ignore close encounters
// size_t close_encounters = 0; // number of close encounters

void integrator_runge_kutta(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real G, double _t, double _t_end, double _dt);

#endif
