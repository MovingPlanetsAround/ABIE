#include "common.h"
#include "integrator_runge_kutta.h"

// vec is the combination of pos+vel
// void integrator_runge_kutta(real *vec, size_t N, real G, real dt, const real *masses) {
void integrate_rk(real *y0, real *dy0, real *masses, real *radii, size_t N, real G, real t, real t_end, real dt) {
    real *k1 = (real *) malloc(6*N*sizeof(real));
    real *k2 = (real *) malloc(6*N*sizeof(real));
    real *k3 = (real *) malloc(6*N*sizeof(real));
    real *k4 = (real *) malloc(6*N*sizeof(real));
    real *vec_tmp = (real *) malloc(6*N*sizeof(real));
    real *vec = (real *) malloc(6*N*sizeof(real));

    for (size_t i = 0; i < 3 * N; i++) vec[i] = y0[i];
    for (size_t i = 3 * N; i < 6 * N; i++) vec[i] = dy0[i - 3 * N];

    while (t < t_end) {
        ode_n_body_first_order(vec, N, G, masses, k1);

        for (size_t i = 0; i < 6 * N; i++) {
            vec_tmp[i] = vec[i] + 0.5 * dt * k1[i];
        }
        ode_n_body_first_order(vec_tmp, N, G, masses, k2);

        for (size_t i = 0; i < 6 * N; i++) {
            vec_tmp[i] = vec[i] + 0.5 * dt * k2[i];
        }
        ode_n_body_first_order(vec_tmp, N, G, masses, k3);

        for (size_t i = 0; i < 6 * N; i++) {
            vec_tmp[i] = vec[i] + dt * k3[i];
        }
        ode_n_body_first_order(vec_tmp, N, G, masses, k4);

        // advance the state
        for (size_t i = 0; i < 6 * N; i++) {
            vec[i] += (dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0);
        }
        t += dt;
    }

    for (size_t i = 0; i < 3 * N; i++) y0[i] = vec[i];
    for (size_t i = 3 * N; i < 6 * N; i++) dy0[i - 3 * N] = vec[i];
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(vec_tmp);
    return;
}

void integrator_runge_kutta(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real G, double _t, double _t_end, double _dt) {
    // allocation
    real t = (real) _t;
    real t_end = (real) _t_end;
    dt = (real) _dt;

    // integrate
    integrate_rk(pos, vel, m_vec, r_vec, N, G, t, t_end, dt);

}
