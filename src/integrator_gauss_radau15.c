#include "common.h"
#include "integrator_gauss_radau15.h"

// Global variables
size_t nh = 8;
size_t dim = 0;
int initialized = 0;

real initial_time_step(const real* y0, real* dy0, real G, real * masses, real *radii, size_t nbodies){

    int p = 15;
    real d0, d1, d2, dt, dt0, dt1;
    real *f0 = (real *) malloc(3 * nbodies * sizeof(real));
    real *F1 = (real *) malloc(3 * nbodies * sizeof(real));
    real *y1 = (real *) malloc(3 * nbodies * sizeof(real));
    real *dy1 = (real *) malloc(3 * nbodies * sizeof(real));
    //###########   ESTIMATE INITIAL STEP SIZE
    //# Compute scaling
    //# sc =  abs(y0)*epsb
    //# Evaluate function

    calculate_accelerations(y0, dy0, nbodies, G, masses, radii, f0);
    d0 = vector_max_abs(y0, nbodies);
    d1 = vector_max_abs(f0, nbodies);

    if (d0 < 1e-5 || d1 < 1e-5){
        dt0 = 1e-6;
    } else {
        dt0 = 0.01 * (d0 / d1);
    }

    // # Perform one Euler step
    for (size_t i = 0; i < 3 * nbodies; i++) {
        y1[i] = y0[i] + dt0 * dy0[i];
        dy1[i] = dy0[i] + dt0 * f0[i];
    }
    // # Call function
    calculate_accelerations(y1, dy1, nbodies, G, masses, radii, F1);
    d2 = -DBL_MAX;
    for (size_t i = 0; i < nbodies * 3; i++) {
        if (d2 < fabs(F1[i] - f0[i])) d2 = fabs(F1[i] - f0[i]);
    }
    d2 = d2 / dt0;

    if (fmax(d1, d2) <= 1e-15) {
        dt1 = fmax(1e-6, dt0 * 1e-3);
    } else {
        dt1 = pow(0.01 / fmax(d1, d2), (1.0 / (p + 1)));
    }

    dt = fmin(100 * dt0, dt1);
    free(f0); free(F1); free(y1); free(dy1);
    return dt;
}

void approx_pos(const real y1[], const real dy1[], const real F1[], real h, real b[][dim], size_t N, real T, real y[]){
    for (size_t i = 0; i < 3 * N; i++) {
        y[i] = y1[i] + T * h * (dy1[i] + T * h * (F1[i] + h * (b[0][i] / 0.3e1 + h * (b[1][i] / 0.6e1 + h * (b[2][i] / 0.10e2 + h * (b[3][i] / 0.15e2 + h * (b[4][i] / 0.21e2 + h * (b[5][i] / 0.28e2 + h * b[6][i] / 0.36e2))))))) / 0.2e1);
    }
    return;
}

void approx_vel(const real dy1[], const real F1[], real h, real b[][dim], size_t N, real T, real dy[]){
    for (size_t i = 0; i < 3 * N; i++) {
        dy[i] = dy1[i] + T * h * (F1[i] + h * (b[0][i] / 0.2e1 + h * (b[1][i] / 0.3e1 + h * (b[2][i] / 0.4e1 + h * (b[3][i] / 0.5e1 + h * (b[4][i] / 0.6e1 + h * (b[5][i] / 0.7e1 + h * b[6][i] / 0.8e1)))))));
    }
    return;
}

void compute_gs(real ddys[][dim], int ih, size_t N, real g[][dim]) {

        const real *F1 = ddys[0];
        const real *F2 = ddys[1];
        const real *F3 = ddys[2];
        const real *F4 = ddys[3];
        const real *F5 = ddys[4];
        const real *F6 = ddys[5];
        const real *F7 = ddys[6];
        const real *F8 = ddys[7];

        // # Update g's with accelerations
        for (size_t i = 0; i < 3 * N; i++) {
            if (ih == 1) {
                g[0][i] = (F2[i] - F1[i]) * rs[1][0];
            } else if (ih == 2) {
                g[0][i] = (F2[i] - F1[i]) * rs[1][0];
                g[1][i] = ((F3[i] - F1[i]) * rs[2][0] - g[0][i]) * rs[2][1];
            } else if (ih == 3) {
                g[0][i] = (F2[i] - F1[i]) * rs[1][0];
                g[1][i] = ((F3[i] - F1[i]) * rs[2][0] - g[0][i]) * rs[2][1];
                g[2][i] = (((F4[i] - F1[i]) * rs[3][0] - g[0][i]) * rs[3][1] - g[1][i]) * rs[3][2];
            } else if (ih == 4) {
                g[0][i] = (F2[i] - F1[i]) * rs[1][0];
                g[1][i] = ((F3[i] - F1[i]) * rs[2][0] - g[0][i]) * rs[2][1];
                g[2][i] = (((F4[i] - F1[i]) * rs[3][0] - g[0][i]) * rs[3][1] - g[1][i]) * rs[3][2];
                g[3][i] = ((((F5[i] - F1[i]) * rs[4][0] - g[0][i]) * rs[4][1] - g[1][i]) * rs[4][2] - g[2][i]) * rs[4][3];
            } else if (ih == 5) {
                g[0][i] = (F2[i] - F1[i]) * rs[1][0];
                g[1][i] = ((F3[i] - F1[i]) * rs[2][0] - g[0][i]) * rs[2][1];
                g[2][i] = (((F4[i] - F1[i]) * rs[3][0] - g[0][i]) * rs[3][1] - g[1][i]) * rs[3][2];
                g[3][i] = ((((F5[i] - F1[i]) * rs[4][0] - g[0][i]) * rs[4][1] - g[1][i]) * rs[4][2] - g[2][i]) * rs[4][3];
                g[4][i] = (((((F6[i] - F1[i]) * rs[5][0] - g[0][i]) * rs[5][1] - g[1][i]) * rs[5][2] - g[2][i]) * rs[5][3] - g[3][i]) * rs[5][4];
            } else if (ih == 6) {
                g[0][i] = (F2[i] - F1[i]) * rs[1][0];
                g[1][i] = ((F3[i] - F1[i]) * rs[2][0] - g[0][i]) * rs[2][1];
                g[2][i] = (((F4[i] - F1[i]) * rs[3][0] - g[0][i]) * rs[3][1] - g[1][i]) * rs[3][2];
                g[3][i] = ((((F5[i] - F1[i]) * rs[4][0] - g[0][i]) * rs[4][1] - g[1][i]) * rs[4][2] - g[2][i]) * rs[4][3];
                g[4][i] = (((((F6[i] - F1[i]) * rs[5][0] - g[0][i]) * rs[5][1] - g[1][i]) * rs[5][2] - g[2][i]) * rs[5][3] - g[3][i]) * rs[5][4];
                g[5][i] = ((((((F7[i] - F1[i]) * rs[6][0] - g[0][i]) * rs[6][1] - g[1][i]) * rs[6][2] - g[2][i]) * rs[6][3] - g[3][i]) * rs[6][4] - g[4][i]) * rs[6][5];
            } else if (ih == 7) {
                g[0][i] = (F2[i] - F1[i]) * rs[1][0];
                g[1][i] = ((F3[i] - F1[i]) * rs[2][0] - g[0][i]) * rs[2][1];
                g[2][i] = (((F4[i] - F1[i]) * rs[3][0] - g[0][i]) * rs[3][1] - g[1][i]) * rs[3][2];
                g[3][i] = ((((F5[i] - F1[i]) * rs[4][0] - g[0][i]) * rs[4][1] - g[1][i]) * rs[4][2] - g[2][i]) * rs[4][3];
                g[4][i] = (((((F6[i] - F1[i]) * rs[5][0] - g[0][i]) * rs[5][1] - g[1][i]) * rs[5][2] - g[2][i]) * rs[5][3] - g[3][i]) * rs[5][4];
                g[5][i] = ((((((F7[i] - F1[i]) * rs[6][0] - g[0][i]) * rs[6][1] - g[1][i]) * rs[6][2] - g[2][i]) * rs[6][3] - g[3][i]) * rs[6][4] - g[4][i]) * rs[6][5];
                g[6][i] = (((((((F8[i] - F1[i]) * rs[7][0] - g[0][i]) * rs[7][1] - g[1][i]) * rs[7][2] - g[2][i]) * rs[7][3] - g[3][i]) * rs[7][4] - g[4][i]) * rs[7][5] - g[5][i]) * rs[7][6];
            }
        }
        return;
}

void compute_bs_from_gs(real g[][dim], int ih, size_t N, real b[][dim]){
    if (ih == 1) {
        for (size_t i = 0; i < 3 * N; i++) {
            b[0][i] = cs[0][0]*g[0][i] + cs[1][0]*g[1][i] + cs[2][0]*g[2][i] + cs[3][0]*g[3][i] + cs[4][0]*g[4][i] + cs[5][0]*g[5][i] + cs[6][0]*g[6][i];
        }
    } else if (ih == 2) {
        for (size_t i = 0; i < 3 * N; i++) {
            b[0][i] = cs[0][0]*g[0][i] + cs[1][0]*g[1][i] + cs[2][0]*g[2][i] + cs[3][0]*g[3][i] + cs[4][0]*g[4][i] + cs[5][0]*g[5][i] + cs[6][0]*g[6][i];
            b[1][i] =                  + cs[1][1]*g[1][i] + cs[2][1]*g[2][i] + cs[3][1]*g[3][i] + cs[4][1]*g[4][i] + cs[5][1]*g[5][i] + cs[6][1]*g[6][i];
        }
    } else if (ih == 3) {
        for (size_t i = 0; i < 3 * N; i++) {
            b[0][i] = cs[0][0]*g[0][i] + cs[1][0]*g[1][i] + cs[2][0]*g[2][i] + cs[3][0]*g[3][i] + cs[4][0]*g[4][i] + cs[5][0]*g[5][i] + cs[6][0]*g[6][i];
            b[1][i] =                  + cs[1][1]*g[1][i] + cs[2][1]*g[2][i] + cs[3][1]*g[3][i] + cs[4][1]*g[4][i] + cs[5][1]*g[5][i] + cs[6][1]*g[6][i];
            b[2][i] =                                     + cs[2][2]*g[2][i] + cs[3][2]*g[3][i] + cs[4][2]*g[4][i] + cs[5][2]*g[5][i] + cs[6][2]*g[6][i];
        }
    } else if (ih == 4) {
        for (size_t i = 0; i < 3 * N; i++) {
            b[0][i] = cs[0][0]*g[0][i] + cs[1][0]*g[1][i] + cs[2][0]*g[2][i] + cs[3][0]*g[3][i] + cs[4][0]*g[4][i] + cs[5][0]*g[5][i] + cs[6][0]*g[6][i];
            b[1][i] =                  + cs[1][1]*g[1][i] + cs[2][1]*g[2][i] + cs[3][1]*g[3][i] + cs[4][1]*g[4][i] + cs[5][1]*g[5][i] + cs[6][1]*g[6][i];
            b[2][i] =                                     + cs[2][2]*g[2][i] + cs[3][2]*g[3][i] + cs[4][2]*g[4][i] + cs[5][2]*g[5][i] + cs[6][2]*g[6][i];
            b[3][i] =                                                          cs[3][3]*g[3][i] + cs[4][3]*g[4][i] + cs[5][3]*g[5][i] + cs[6][3]*g[6][i];
        }
    } else if (ih == 5) {
        for (size_t i = 0; i < 3 * N; i++) {
            b[0][i] = cs[0][0]*g[0][i] + cs[1][0]*g[1][i] + cs[2][0]*g[2][i] + cs[3][0]*g[3][i] + cs[4][0]*g[4][i] + cs[5][0]*g[5][i] + cs[6][0]*g[6][i];
            b[1][i] =                  + cs[1][1]*g[1][i] + cs[2][1]*g[2][i] + cs[3][1]*g[3][i] + cs[4][1]*g[4][i] + cs[5][1]*g[5][i] + cs[6][1]*g[6][i];
            b[2][i] =                                     + cs[2][2]*g[2][i] + cs[3][2]*g[3][i] + cs[4][2]*g[4][i] + cs[5][2]*g[5][i] + cs[6][2]*g[6][i];
            b[3][i] =                                                          cs[3][3]*g[3][i] + cs[4][3]*g[4][i] + cs[5][3]*g[5][i] + cs[6][3]*g[6][i];
            b[4][i] =                                                                             cs[4][4]*g[4][i] + cs[5][4]*g[5][i] + cs[6][4]*g[6][i];
        }
    } else if (ih == 6) {
        for (size_t i = 0; i < 3 * N; i++) {
            b[0][i] = cs[0][0]*g[0][i] + cs[1][0]*g[1][i] + cs[2][0]*g[2][i] + cs[3][0]*g[3][i] + cs[4][0]*g[4][i] + cs[5][0]*g[5][i] + cs[6][0]*g[6][i];
            b[1][i] =                  + cs[1][1]*g[1][i] + cs[2][1]*g[2][i] + cs[3][1]*g[3][i] + cs[4][1]*g[4][i] + cs[5][1]*g[5][i] + cs[6][1]*g[6][i];
            b[2][i] =                                     + cs[2][2]*g[2][i] + cs[3][2]*g[3][i] + cs[4][2]*g[4][i] + cs[5][2]*g[5][i] + cs[6][2]*g[6][i];
            b[3][i] =                                                          cs[3][3]*g[3][i] + cs[4][3]*g[4][i] + cs[5][3]*g[5][i] + cs[6][3]*g[6][i];
            b[4][i] =                                                                             cs[4][4]*g[4][i] + cs[5][4]*g[5][i] + cs[6][4]*g[6][i];
            b[5][i] =                                                                                                cs[5][5]*g[5][i] + cs[6][5]*g[6][i];
        }
    } else if (ih == 7) {
        for (size_t i = 0; i < 3 * N; i++) {
            b[0][i] = cs[0][0]*g[0][i] + cs[1][0]*g[1][i] + cs[2][0]*g[2][i] + cs[3][0]*g[3][i] + cs[4][0]*g[4][i] + cs[5][0]*g[5][i] + cs[6][0]*g[6][i];
            b[1][i] =                  + cs[1][1]*g[1][i] + cs[2][1]*g[2][i] + cs[3][1]*g[3][i] + cs[4][1]*g[4][i] + cs[5][1]*g[5][i] + cs[6][1]*g[6][i];
            b[2][i] =                                     + cs[2][2]*g[2][i] + cs[3][2]*g[3][i] + cs[4][2]*g[4][i] + cs[5][2]*g[5][i] + cs[6][2]*g[6][i];
            b[3][i] =                                                          cs[3][3]*g[3][i] + cs[4][3]*g[4][i] + cs[5][3]*g[5][i] + cs[6][3]*g[6][i];
            b[4][i] =                                                                             cs[4][4]*g[4][i] + cs[5][4]*g[5][i] + cs[6][4]*g[6][i];
            b[5][i] =                                                                                                cs[5][5]*g[5][i] + cs[6][5]*g[6][i];
            b[6][i] =                                                                                                                   cs[6][6]*g[6][i];
        }
    }
    return;
}


void refine_bs(real b[][dim], real q, real E[][dim], size_t N){
    real bd[nh - 1][3 * N];
    static int inited = 0;
    if (inited != 0){
        for (size_t i = 0; i < 3 * N; i++) {
            for (size_t j = 0; j < 7; j++) {
                bd[j][i] = b[j][i] - E[j][i];
            }
        }
    } else {
        for (size_t i = 0; i < 3 * N; i++) {
            for (size_t j = 0; j < 7; j++) {
                bd[j][i] = 0.0;
            }
        }
    }

    real q2 = q * q;
    real q3 = q2 * q;
    real q4 = q2 * q2;
    real q5 = q2 * q3;
    real q6 = q3 * q3;
    real q7 = q2 * q5;

    for (size_t i = 0; i < 3 * N; i++) {
        E[0][i] = q * (b[6][i] * 7.0 + b[5][i] * 6.0 + b[4][i] * 5.0 + b[3][i] * 4.0 + b[2][i] * 3.0 + b[1][i] * 2.0 + b[0][i]);
        E[1][i] = q2 * (b[6][i] * 21.0 + b[5][i] * 15.0 + b[4][i] * 10.0 + b[3][i] * 6.0 + b[2][i] * 3.0 + b[1][i]);
        E[2][i] = q3 * (b[6][i] * 35.0 + b[5][i] * 20.0 + b[4][i] * 10.0 + b[3][i] * 4.0 + b[2][i]);
        E[3][i] = q4 * (b[6][i] * 35.0 + b[5][i] * 15.0 + b[4][i] * 5.0 + b[3][i]);
        E[4][i] = q5 * (b[6][i] * 21.0 + b[5][i] * 6.0 + b[4][i]);
        E[5][i] = q6 * (b[6][i] * 7.0 + b[5][i]);
        E[6][i] = q7 * b[6][i];
    }

    for (size_t i = 0; i < 3 * N; i++) {
        for (size_t j = 0; j < 7; j++) {
            b[j][i] = E[j][i] + bd[j][i];
        }
    }
    return;
}


size_t integrator_gauss_radau15(real *pos, real *vel, real *m_vec, real *r_vec, size_t N, real _G, real _t, real _t_end, real _dt) {
    // # Dimension of the system
    dim = 3 * N;

    // allocate
    real *y0 = (real *) malloc(dim * sizeof(real));
    real *dy0 = (real *) malloc(dim * sizeof(real));
    real *masses = (real *) malloc(N * sizeof(real));
    real *ddy0 = (real *) malloc(dim * sizeof(real));
    real *y = (real *) malloc(dim * sizeof(real));
    real *dy = (real *) malloc(dim * sizeof(real));
    real *db6 = (real *) malloc(dim * sizeof(real));
    real *ddy = (real *) malloc(dim * sizeof(real));
    real bs0[nh - 1][dim];
    real bs[nh - 1][dim];
    real g[nh - 1][dim];
    real E[nh - 1][dim];
    real ddys[nh][dim];
    real h = (real) _dt; // timestep

    // type casting
    real G = _G;
    real t = _t;
    real t_end = _t_end;
    size_t integrator_flag = 0;

    for (size_t i = 0; i < 3 * N; i++) {
        y0[i] = pos[i];
        dy0[i] = vel[i];
    }
    for (size_t i = 0; i < N; i++) masses[i] = m_vec[i];

    // # Initialize
    for (size_t j = 0; j < nh - 1; j++) {
        for (size_t i = 0; i < dim; i++) {
            bs0[j][i] = 0.0;
            bs[j][i] = 0.0;
            g[j][i] = 0.0;
            E[j][i] = 0.0;
        }
    }
    for (size_t j = 0; j < nh; j++) {
        for (size_t i = 0; i < dim; i++) ddys[j][i] = 0.0;
    }
    for (size_t i = 0; i < dim; i++) {
        db6[i] = 0.0;
        ddy[i] = 0.0;
    }

    calculate_accelerations(y0, dy0, N, G, masses, r_vec, ddy0);

    // # Initial time step
    h = initial_time_step(y0, dy0, G, masses, r_vec, N);

    int advance_step = 1;
    int step_loop_count = 0;
    int step_loop_max = 100;
    int warning_msg_printed = 0;
    while(advance_step){
        // # Variable number of iterations in PC
        for (size_t ipc = 0; ipc < 12; ipc++) {
            for (size_t j = 0; j < nh; j++) {
                for (size_t i = 0; i < 3 * N; i++) {
                    ddys[j][i] = 0;
                }
            }
            // # Advance along the Radau sequence
            for (size_t ih = 0; ih < nh; ih++) {
                // # Estimate position and velocity with bs0 and current h
                approx_pos(y0, dy0, ddy0, hs[ih], bs, N, h, y);
                approx_vel(dy0, ddy0, hs[ih], bs, N, h, dy);
                // # Evaluate force function and store
                calculate_accelerations(y, dy, N, G, masses, r_vec, ddys[ih]);
                compute_gs(ddys, ih, N, g);
                compute_bs_from_gs(g, ih, N, bs);
            }

            for (size_t i = 0; i < dim; i++) db6[i] = bs[nh - 2][i] - bs0[nh - 2][i];

            if (vector_max_abs(db6, dim) / vector_max_abs(ddys[nh - 1], dim) < tolpc) break;
            for (size_t j = 0; j < nh - 1; j++) {
                for (size_t i = 0; i < dim; i++) bs0[j][i] = bs[j][i];
            }
            // if (integrator_flag > 0) break;
        }

        // ################# ADVANCE SOLUTION

        approx_pos(y0, dy0, ddy0, 1., bs, N, h, y);
        approx_vel(dy0, ddy0, 1., bs, N, h, dy);
        calculate_accelerations(y, dy, N, G, masses, r_vec, ddy);

        // ################## COMPUTE STEP-SIZE
        // # Estimate relative error
        real estim_b6 = vector_max_abs(bs[nh - 2], dim) / vector_max_abs(ddy, dim);
        real err = pow(estim_b6 / epsb, exponent);
        real dtreq = h / err;

        // # Accept the step
        if (err <= 1 || step_loop_count > step_loop_max) {
            t += h;
            t_global = t;
            step_loop_count = 0;
            // param[0] = t;
            // param[1] = h;
            // param[2] = close_encounters;
            if (t > t_end){
                h = t_end - t;
                advance_step = 0;
            }
            // # Update step
            for (size_t i = 0; i < dim; i++) {
                y0[i] = y[i];
                dy0[i] = dy[i];
                ddy0[i] = ddy[i];
            }
            for (size_t j = 0; j < nh - 1; j++) {
                for (size_t i = 0; i < dim; i++) bs0[j][i] = bs[j][i];
            }

            refine_bs(bs, dtreq / h, E, N);
            integrator_flag = check_collisions_close_encounters(y, r_vec, N, t);
            //if (integrator_flag > 0) return integrator_flag; // return if collision or close encounters are detected
            if (integrator_flag > 0) break; // break the while(advance_step) loop, but still allow the subsequent clean-up process
        } else{
            // if the timestep is not accepted, record the times of rejection
            step_loop_count += 1;
        }
        if (dtreq / h > 1.0 / fac){
            h /= fac;
        } else if (dtreq < 1.e-12){
            h *= fac;
        } else {
            h = dtreq;
        }
        // printf("h = %g\n", h);
        if (h < h_min){
            if (warning_msg_printed == 0){
#ifdef LONGDOUBLE
                printf("Warning! Timestep %Lg being too small! Imposing a minimum timestep of %Lg, t = %Lg\n", h, h_min, t);
#else
                printf("Warning! Timestep %g being too small! Imposing a minimum timestep of %g, t = %g\n", h, h_min, t);
#endif
                warning_msg_printed = 1; // suppress further warning
            }
            h = h_min;
        } // if (h < h_min)
    } // while(advance_step)

    // type casting
    for (size_t i = 0; i < 3 * N; i++) {
        pos[i] = (double) y0[i];
        vel[i] = (double) dy[i];
    }
    for (size_t i = 0; i < N; i++) m_vec[i] = (double) masses[i];
    free(y0); free(dy0); free(masses); free(ddy0); 
    free(y);  free(dy);  free(db6);    free(ddy);

    return integrator_flag;
}
