#include "common.h"

size_t ode_n_body_first_order(real *vec, size_t N, real G, const real *masses, real *dxdt) {
    real x, y, z;
    real dx, dy, dz;
    real ax, ay, az;
    real rel_sep3;
    real rel_sep2;
    real GM;
    for (size_t i = 0; i < 3 * N; i++){
        dxdt[i] = vec[3*N+i];
    }
    for (size_t j = 0; j < N; j++) {
        x = vec[j * 3];
        y = vec[j * 3 + 1];
        z = vec[j * 3 + 2];
        ax = 0.0;
        ay = 0.0;
        az = 0.0;
        for (size_t k = 0; k < N; k++) {
            if ((j == k) || (masses[k] == 0)) continue;
            GM = G * masses[k];
            dx = x - vec[k*3];
            dy = y - vec[k*3+1];
            dz = z - vec[k*3+2];
            rel_sep2 = dx * dx + dy * dy + dz * dz;
            rel_sep3 = sqrt(rel_sep2) * rel_sep2;
            ax -= (GM * dx / rel_sep3);
            ay -= (GM * dy / rel_sep3);
            az -= (GM * dz / rel_sep3);
        }
        dxdt[(N+j)*3] = ax;
        dxdt[(N+j)*3+1] = ay;
        dxdt[(N+j)*3+2] = az;
    }
    return EXIT_NORMAL;
}

size_t ode_n_body_second_order(const real vec[], size_t N, real G, const real masses[], const real radii[], real acc[]) {
    real x, y, z;
    real dx, dy, dz;
    real ax, ay, az;
    real GM;

    // Calculate the combined accelerations onto particle j
    // i.e. j is the sink, k is the source

    for (size_t j = 0; j < N; j++){
        if (masses[j] < 0.0) {
            // if the mass is negative, the particle is deleted
            continue;
        }
        x = vec[j * 3];
        y = vec[j * 3 + 1];
        z = vec[j * 3 + 2];
        ax = 0.0;
        ay = 0.0;
        az = 0.0;
        for (size_t k = 0; k < N; k++) {
            if (j == k || masses[k] <= 0.0) continue;
            GM = G * masses[k];
            dx = x - vec[k * 3];
            dy = y - vec[k * 3 + 1];
            dz = z - vec[k * 3 + 2];
            real rel_sep2 = dx * dx + dy * dy + dz * dz;
            real rel_sep = sqrt(rel_sep2);
            real rel_sep3 = rel_sep * rel_sep2;
            ax -= (GM * dx / rel_sep3);
            ay -= (GM * dy / rel_sep3);
            az -= (GM * dz / rel_sep3);
        }
        acc[j * 3] = ax;
        acc[j * 3 + 1] = ay;
        acc[j * 3 + 2] = az;
    }
    return EXIT_NORMAL;
}

/*
 * calculate the accelerations due to N bodies.
*/
size_t calculate_accelerations(const real pos[], const real vel[], size_t N, real G, const real masses[], const real radii[], real acc[]) {

#ifdef GPU
    if (sim.devID >= 0) {
        // Use GPU to carry out the force calculation when N is large
        ode_n_body_second_order_gpu(pos, N, G, masses, radii, acc);
    } else {
        // Switch to CPU when N is small
        ode_n_body_second_order(pos, N, G, masses, radii, acc);
    }
#elif SAPPORO
    ode_n_body_second_order_sapporo(pos, N, G, masses, radii, acc);
#else
    ode_n_body_second_order(pos, N, G, masses, radii, acc);
#endif

    // calculate additional accelerations
    // vel[] is passed to the function because the additional force can be velocity-dependenet
    // C_global is the speed of light
#ifdef LONGDOUBLE
    calculate_additional_forces(pos, vel, N, G, (double)sim.C_global, masses, radii, acc);
#else
    calculate_additional_forces(pos, vel, N, G, sim.C_global, masses, radii, acc);
#endif

    // add up externally calculated accelerations
    if ((sim.ENABLE_EXT_ACC > 0) && (sim.ext_acc_global != NULL)) {
        for (size_t i = 0; i < 3 * N; i++) {
            acc[i] += sim.ext_acc_global[i];
        }
    }
    return EXIT_NORMAL;
}

size_t check_collisions_close_encounters(const real *vec, const real radii[], size_t N, real t) {
    real x, y, z;
    real dx, dy, dz;
    for (size_t j = 0; j < N; j++){
        x = vec[j * 3];
        y = vec[j * 3 + 1];
        z = vec[j * 3 + 2];
        for (size_t k = 0; k < N; k++) {
            if (j == k) continue;
            dx = x - vec[k * 3];
            dy = y - vec[k * 3 + 1];
            dz = z - vec[k * 3 + 2];
            real rel_sep2 = dx * dx + dy * dy + dz * dz;
            real rel_sep = sqrt(rel_sep2);
            real r = radii[j] + radii[k];

            // close encounter detection
            if (rel_sep <= sim.close_encounter_distance && (j < k)){
                if (sim.buf_ce_events != NULL) {
                    sim.buf_ce_events[(4 * sim.n_close_encounters) % (4 * sim.MAX_N_CE)] = t;
                    sim.buf_ce_events[(4 * sim.n_close_encounters + 1) % (4 * sim.MAX_N_CE)] = j;
                    sim.buf_ce_events[(4 * sim.n_close_encounters + 2) % (4 * sim.MAX_N_CE)] = k;
                    sim.buf_ce_events[(4 * sim.n_close_encounters + 3)  % (4 * sim.MAX_N_CE)] = rel_sep;
                }
                sim.n_close_encounters += 1;
            }

            // collision detection
            if ((r > 0) && (rel_sep <= r) && (j < k)) {
                if (sim.buf_collision_events != NULL) {
                    sim.buf_collision_events[(4 * sim.n_collisions) % (4 * sim.MAX_N_COLLISIONS)] = t;
                    sim.buf_collision_events[(4 * sim.n_collisions + 1) % (4 * sim.MAX_N_COLLISIONS)] = j;
                    sim.buf_collision_events[(4 * sim.n_collisions + 2) % (4 * sim.MAX_N_COLLISIONS)] = k;
                    sim.buf_collision_events[(4 * sim.n_collisions + 3)  % (4 * sim.MAX_N_COLLISIONS)] = rel_sep;
                }
                sim.n_collisions += 1;
            }
        }
    }
    if ((sim.MAX_N_CE) > 0 && (sim.n_close_encounters >= sim.MAX_N_CE)) return EXIT_MAX_N_CE_EXCEEDED;
    else if ((sim.MAX_N_COLLISIONS > 0) && (sim.n_collisions >= sim.MAX_N_COLLISIONS)) return EXIT_MAX_N_COLLISIONS_EXCEEDED;
    else return EXIT_NORMAL;
}

real *vec_scalar_op(const real *vec, real scalar, size_t N, char op) {
    real *res = (real *) malloc(6*N*sizeof(real));
    for (size_t i = 0; i < N; i++) {
        if (op == '+') {
            res[i] = vec[i] + scalar;
        } else if (op == '-') {
            res[i] = vec[i] - scalar;
        } else if (op == '*') {
            res[i] = vec[i] * scalar;
        }  else if (op == '/') {
            res[i] = vec[i] / scalar;
        }
    }
    return res;
}

real *vec_vec_op(const real *vec1, real *vec2, size_t N, char op) {
    real *res = (real *) malloc(6*N*sizeof(real));
    for (size_t i = 0; i < N; i++) {
        if (op == '+') {
            res[i] = vec1[i] + vec2[i];
        } else if (op == '-') {
            res[i] = vec1[i] - vec2[i];
        } else if (op == '*') {
            res[i] = vec1[i] * vec2[i];
        } else if (op == '/') {
            res[i] = vec1[i] / vec2[i];
        }
    }
    return res;
}

real vector_max_abs(const real *vec, size_t N) {
    real max_val = -DBL_MAX;
    for (size_t i = 0; i < N; i++) {
        if (max_val < fabs(vec[i])) max_val = fabs(vec[i]);
    }
    return max_val;
}

double get_model_time() {
    return (double) sim.t_global;
}

real vector_norm(const real *vec, size_t N) {
    real sum = 0.0;
    for (size_t i = 0; i < N; i++) sum += (vec[i] * vec[i]);
    return sqrt(sum);
}

real sign(real x) {
    return (x > 0) - (x < 0);
}

real dot(const real *vec1, const real *vec2) {
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

real cross_norm(const real *vec1, const real *vec2) {
    real c0 = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    real c1 = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    real c2 = vec1[0] * vec2[1] - vec1[1] * vec2[0];
    return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}

int code_inited = 0;
int initialize_code(double _G, double _C, size_t _N_MAX, size_t _MAX_N_CE, size_t _MAX_N_COLLISIONS, int deviceID) {
    if (code_inited > 0) return 0;
    printf("Initializing the code..., deviceID = %d", deviceID);
    // define constants and flags
    sim.MAX_N_CE = _MAX_N_CE;
    sim.MAX_N_COLLISIONS = _MAX_N_COLLISIONS;
    // EXIT_MAX_N_CE_EXCEEDED = 1;
    // EXIT_MAX_N_COLLISIONS_EXCEEDED = 2;
    // Assume no externally calculated forces, unless the set_ext_acc() function is invoked
    if (sim.ENABLE_EXT_ACC <= 0) sim.ENABLE_EXT_ACC = 0;

    sim.G_global = (real) _G;
    sim.C_global = (real) _C;

    // set the devID to -1 until GPU is found and initialized
    sim.devID = -1;

    // allocate
    // TODO: reallocate the memory if N changes
    if (sim.pos_global == NULL) sim.pos_global = (real *) malloc(3 * _N_MAX * sizeof(real));
    if (sim.vel_global == NULL) sim.vel_global = (real *) malloc(3 * _N_MAX * sizeof(real));
    if (sim.ext_acc_global == NULL) {
        sim.ext_acc_global = (real *) malloc(3 * _N_MAX * sizeof(real));
        for (size_t i = 0; i < 3 * _N_MAX; i++) sim.ext_acc_global[i] = 0.0;
    }
    if (sim.m_vec_global == NULL) sim.m_vec_global = (real *) malloc(_N_MAX * sizeof(real));
    if (sim.r_vec_global == NULL) sim.r_vec_global = (real *) malloc(_N_MAX * sizeof(real));

    // For conveniece access in the python interface, these buffers are allocated as double always
    if (sim.MAX_N_CE > 0) sim.buf_ce_events = (real *) malloc(4 * sim.MAX_N_CE * sizeof(real));
    if (sim.MAX_N_COLLISIONS > 0) sim.buf_collision_events = (real *) malloc(4 * sim.MAX_N_COLLISIONS * sizeof(real));

    // initialize variables
    sim.n_close_encounters = 0;
    // buf_collision_events = NULL;
    for (size_t i = 0; i < 4 * sim.MAX_N_CE; i++) sim.buf_ce_events[i] = 0.0;
    for (size_t i = 0; i < 4 * sim.MAX_N_COLLISIONS; i++) sim.buf_collision_events[i] = 0.0;

#ifdef GPU
    gpu_init(_N_MAX, deviceID);
#endif

#ifdef SAPPORO
    initialize_sapporo();
#endif
    code_inited = 1;
    printf("Initialized.\n");

    return 0;
}

void set_close_encounter_distance(double d) {
    sim.close_encounter_distance = (real) d;
}

double get_close_encounter_distance() {
    return (double) sim.close_encounter_distance;
}

void get_close_encounter_buffer(double *buf_ce) {
    for (size_t i = 0; i < 4 * sim.MAX_N_CE; i++) {
        buf_ce[i] = (double) sim.buf_ce_events[i];
    }
}

void get_collision_buffer(double *buf_collision) {
    for (size_t i = 0; i < 4 * sim.MAX_N_COLLISIONS; i++) {
        buf_collision[i] = (double) sim.buf_collision_events[i];
    }
}

void reset_close_encounter_buffer() {
    // this should be called after the python interface finishes handling a close encounter exception
    sim.n_close_encounters = 0;
    for (size_t i = 0; i < 4 * sim.MAX_N_CE; i++) sim.buf_ce_events[i] = 0.0;
}

void reset_collision_buffer() {
    // this should be called after the python interface finishes handling a collision exception
    sim.n_collisions = 0;
    for (size_t i = 0; i < 4 * sim.MAX_N_COLLISIONS; i++) sim.buf_collision_events[i] = 0.0;
}

void set_state(double *pos_vec, double *vel_vec, double *m_vec, double *r_vec, size_t N, double G, double C){
    // initialize if the global arrays are not allocated
    // initialize_code(G, C, N, MAX_N_CE, MAX_N_COLLISIONS);

    // copy the data from python to the global array with type casting
    for (size_t i = 0; i < 3 * N; i++) {
        sim.pos_global[i] = (real) pos_vec[i];
        sim.vel_global[i] = (real) vel_vec[i];
    }
    for (size_t i = 0; i < N; i++) {
        sim.m_vec_global[i] = (real) m_vec[i];
        sim.r_vec_global[i] = (real) r_vec[i];
    }
    sim.N_global = (size_t) N;
    sim.G_global = (real) G;
    sim.C_global = (real) C;
}

int get_state(double *pos_vec, double *vel_vec, double *m_vec, double *r_vec) {
    // copy the data from the global arrays to the python data space
    for (size_t i = 0; i < 3 * sim.N_global; i++) {
        pos_vec[i] = (double) sim.pos_global[i];
        vel_vec[i] = (double) sim.vel_global[i];
    }
    for (size_t i = 0; i < sim.N_global; i++) {
        m_vec[i] = (double) sim.m_vec_global[i];
        r_vec[i] = (double) sim.r_vec_global[i];
    }
    return (int) sim.N_global;
}

double calculate_energy() {
    real energy = 0.0;
    real d_pos[3];
    for (size_t i = 0; i < sim.N_global; i++) {
        if (sim.m_vec_global[i] == 0) continue;
        // kinetic energy
        energy += (0.5 * sim.m_vec_global[i] * pow(vector_norm(&(sim.vel_global[3 * i]), 3), 2.0));
        // potential energy
        for (size_t j = 0; j < sim.N_global; j++) {
            if ((i == j) || (sim.m_vec_global[j] == 0)) continue;
            d_pos[0] = sim.pos_global[3 * i] - sim.pos_global[3 * j];
            d_pos[1] = sim.pos_global[3 * i + 1] - sim.pos_global[3 * j + 1];
            d_pos[2] = sim.pos_global[3 * i + 2] - sim.pos_global[3 * j + 2];
            energy -= (0.5 * sim.G_global * sim.m_vec_global[i] * sim.m_vec_global[j] / vector_norm(d_pos, 3));
        }
    }
    return (double)energy;
}

/***
 * Plug the additional forces calculated elsewhere into the integrator.
 * WARNING: if the ext_acc[] array is not updated every integration timestep by
 * an external routine, it will be treated as constant during the two updates!
*/
size_t set_additional_forces(size_t N, double ext_acc[]) {
    // set the flag to 10
    sim.ENABLE_EXT_ACC = 10;

    // allocate memory if necessary
    if (sim.ext_acc_global == NULL) {
        printf("Allocating memory for ext_acc_global...\n");
        sim.ext_acc_global = (real *) malloc(3 * N * sizeof(real));
        for (size_t i = 0; i < 3 * N; i++) sim.ext_acc_global[i] = 0.0;
    }

    // set the ext acc with type casting
    for (size_t i = 0; i < (size_t) (3 * N); i++) {
        sim.ext_acc_global[i] = (real) ext_acc[i];
    }
    return 0;
}

int finalize_code() {
    if (sim.pos_global != NULL) free(sim.pos_global);
    if (sim.vel_global != NULL) free(sim.vel_global);
    if (sim.m_vec_global != NULL) free(sim.m_vec_global);
    if (sim.r_vec_global != NULL) free(sim.r_vec_global);
    if (sim.ext_acc_global != NULL) free(sim.ext_acc_global);
    if (sim.buf_ce_events != NULL) free(sim.buf_ce_events);
    if (sim.buf_collision_events != NULL) free(sim.buf_collision_events);
    // t = 0.0;
    // t_end = 0.0;
    //
#ifdef GPU
    gpu_finalize();
#endif

    return 0;
}

int integrator_gr(double t, double t_end, double dt) {
    int ret = (int) integrator_gauss_radau15(sim.pos_global, sim.vel_global, sim.m_vec_global, sim.r_vec_global, sim.N_global, sim.G_global, t, t_end, dt);
    return ret;
}

int integrator_rk(double t, double t_end, double dt) {
    integrator_runge_kutta(sim.pos_global, sim.vel_global, sim.m_vec_global, sim.r_vec_global, sim.N_global, sim.G_global, t, t_end, dt);
    return 0;
}

int integrator_wh(double t, double t_end, double dt) {
    integrator_wisdom_holman(sim.pos_global, sim.vel_global, sim.m_vec_global, sim.r_vec_global, sim.N_global, sim.G_global, t, t_end, dt);
    return 0;
}
