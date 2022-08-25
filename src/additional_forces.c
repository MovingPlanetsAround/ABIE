#include "common.h"

size_t calculate_additional_forces(const real pos[], const real vel[], size_t N, real G, real C, const real masses[], const real radii[], real acc[]) {
    // ext_acc: the additional accelerations
    // put the routine of calculating additional accelerations here
    // for (size_t i = 0; i < 3 * N; i++) acc[i] -= 0.0; // replace this line

    if (C > 0.0) {
        calculate_post_newtonian(pos, vel, N, G, C, masses, radii, acc);
    }

    int calc_j_terms = 1; // TODO: move out the hard-coded config here

    if (calc_j_terms) {
        calculate_j_terms(pos, vel, N, G, C, masses, radii, acc);
    }
    return EXIT_NORMAL;
}

size_t calculate_post_newtonian(const real pos[], const real vel[], size_t N, real G, real C, const real masses[], const real radii[], real acc[]) {
    real x, y, z;
    real dx, dy, dz;

    real vx, vy, vz;
    real dvx, dvy, dvz;

    real ax, ay, az;

    // real r[3], v[3];

    // real CONST_C_LIGHT_P2 = C * C; // CONST_C_LIGHT * CONST_C_LIGHT;
    real CONST_C_LIGHT_PM1 = 1.0 / C;
    real CONST_C_LIGHT_PM2 = CONST_C_LIGHT_PM1 * CONST_C_LIGHT_PM1;
    real CONST_C_LIGHT_PM4 = CONST_C_LIGHT_PM2 * CONST_C_LIGHT_PM2;
    real CONST_C_LIGHT_PM5 = CONST_C_LIGHT_PM1 * CONST_C_LIGHT_PM4;

    real rel_sep1, rel_sep2;
    real rel_sep_div1, rel_sep_div2, rel_sep_div3, rel_sep_div4;
    
    real M_j, M_k, M_tot, nu;
    real GM, GM_tot, GM_C_PM2, GM_P2_C_PM2;
    real GM_P2_RC_PM4;
    real r_dot_v, v_dot_v;

    real c8div5 = (8.0/5.0);
    real c17div3 = (17.0/3.0);
    real FAC0_1PN, FAC1_1PN, FAC2_1PN;
    real FAC0_2PN, FAC1_2PN, FAC2_2PN;
    real FAC0_25PN, FAC1_25PN, FAC2_25PN;

    for (size_t j = 0; j < N; j++){
        x = pos[j * 3];
        y = pos[j * 3 + 1];
        z = pos[j * 3 + 2];
        vx = vel[j * 3];
        vy = vel[j * 3 + 1];
        vz = vel[j * 3 + 2];

        M_j = masses[j];
        ax = 0.0;
        ay = 0.0;
        az = 0.0;
        for (size_t k = 0; k < N; k++) {
            if (j == k) continue;
            M_k = masses[k];
            GM = G * M_k;
            GM_C_PM2 = GM * CONST_C_LIGHT_PM2;
            GM_P2_C_PM2 = GM * GM_C_PM2;

            dx = x - pos[k * 3];
            dy = y - pos[k * 3 + 1];
            dz = z - pos[k * 3 + 2];
            dvx = vx - vel[k * 3];
            dvy = vy - vel[k * 3 + 1];
            dvz = vz - vel[k * 3 + 2];

            r_dot_v = dx * dvx + dy * dvy + dz * dvz;
            v_dot_v = dvx * dvx + dvy * dvy + dvz * dvz;

            rel_sep2 = dx * dx + dy * dy + dz * dz;
            rel_sep1 = sqrt(rel_sep2);

            rel_sep_div1 = 1.0 / rel_sep1;
            rel_sep_div2 = rel_sep_div1 * rel_sep_div1;
            rel_sep_div3 = rel_sep_div1 * rel_sep_div2;
            rel_sep_div4 = rel_sep_div1 * rel_sep_div3;

            // 1PN //
            FAC0_1PN = 4.0 * GM_C_PM2 * r_dot_v * rel_sep_div3;
            FAC1_1PN = 4.0 * GM_P2_C_PM2 * rel_sep_div4;
            FAC2_1PN = -GM_C_PM2 * v_dot_v * rel_sep_div3;

            ax += FAC0_1PN * dvx + FAC1_1PN * dx + FAC2_1PN * dx;
            ay += FAC0_1PN * dvy + FAC1_1PN * dy + FAC2_1PN * dy;
            az += FAC0_1PN * dvz + FAC1_1PN * dz + FAC2_1PN * dz;

            // 2PN //
            GM_P2_RC_PM4 = GM_P2_C_PM2 * CONST_C_LIGHT_PM2 * rel_sep_div4;
            FAC0_2PN = 2.0 * r_dot_v * r_dot_v * rel_sep_div2;
            FAC1_2PN = -9.0 * GM * rel_sep_div1;
            FAC2_2PN = -2.0 * r_dot_v;

            ax += GM_P2_RC_PM4*(FAC0_2PN * dx + FAC1_2PN * dx + FAC2_2PN * dvx);
            ay += GM_P2_RC_PM4*(FAC0_2PN * dy + FAC1_2PN * dy + FAC2_2PN * dvy);
            az += GM_P2_RC_PM4*(FAC0_2PN * dz + FAC1_2PN * dz + FAC2_2PN * dvz);

            // 2.5PN //
            M_tot = M_j + M_k;
            nu = M_j * M_k / (M_tot * M_tot);
            GM_tot = G * M_tot;
            FAC0_25PN = -c8div5 * GM_tot * GM_tot * nu * CONST_C_LIGHT_PM5 * rel_sep_div3;
            FAC1_25PN = v_dot_v + 3.0 * GM_tot * rel_sep_div1;
            FAC2_25PN = -(3.0 * v_dot_v + c17div3 * GM_tot * rel_sep_div1) * r_dot_v * rel_sep_div2;

            ax += FAC0_25PN * (FAC1_25PN * dvx + FAC2_25PN * dx);
            ay += FAC0_25PN * (FAC1_25PN * dvy + FAC2_25PN * dy);
            az += FAC0_25PN * (FAC1_25PN * dvz + FAC2_25PN * dz);

            //printf("test FAC PN %g %g %g %g %g %g %g %g %g\n",FAC0_1PN,FAC1_1PN,FAC2_2PN,FAC0_2PN,FAC1_2PN,FAC2_2PN,FAC0_25PN,FAC1_25PN,FAC2_25PN);

        }
        acc[j * 3] += ax;
        acc[j * 3 + 1] += ay;
        acc[j * 3 + 2] += az;
    }
    return EXIT_NORMAL;
}


void __calculate_grad_i_shat(const real pos[], const real radii[], int i, real3 grad_u_i_shat) {
    // Values for Saturn, according to Gomes-Júnior et al. 2021
    // When working on other planets, these values should be changed accordingly 
    const real J2 = 1.629133249525738e-02;
    const real J3 = 1.494723182852077e-06;
    const real J4 = -9.307138534779719e-04;
    const real J6 = 8.943208329411604E-05;

    real x_i = pos[i * 3]; // x component
    real y_i = pos[i * 3 + 1]; // y component
    real z_i = pos[i * 3 + 2]; // z component
    real r_i = sqrt(x_i * x_i + y_i * y_i + z_i * z_i); // distance to the origin
    real R_i = radii[i];  // radius of the i-th body

    real grad_x_u_i_shat_j2 = - R_i * R_i * J2 * (3.0 / 2.0) * (x_i / pow(r_i, 5) - 5 * x_i * pow(z_i, 2) / pow(r_i, 7));
    real grad_y_u_i_shat_j2 = - R_i * R_i * J2 * (3.0 / 2.0) * (y_i / pow(r_i, 5) - 5 * y_i * pow(z_i, 2) / pow(r_i, 7));
    real grad_z_u_i_shat_j2 = - R_i * R_i * J2 * (3.0 / 2.0) * (z_i / pow(r_i, 5) - 5 * z_i * pow(z_i, 2) / pow(r_i, 7));
    
    
    real grad_x_u_i_shat_j3 = - pow(R_i, 3) * J3 * (5.0 / 2.0) * (3 * x_i * z_i/ pow(r_i, 7) - 7 * x_i * pow(z_i, 3) / pow(r_i, 9));
    real grad_y_u_i_shat_j3 = - pow(R_i, 3) * J3 * (5.0 / 2.0) * (3 * y_i * z_i/ pow(r_i, 7) - 7 * y_i * pow(z_i, 3) / pow(r_i, 9));
    real grad_z_u_i_shat_j3 = - pow(R_i, 3) * J3 * (5.0 / 2.0) * ((3 * z_i * z_i/ pow(r_i, 7) - 7 * z_i * pow(z_i, 3) / pow(r_i, 9)) - (3.0 / 2.0) * (1.0 / pow(r_i, 5) - 5 * pow(z_i, 2) / pow(r_i, 7)));

    real grad_x_u_i_shat_j4 = - pow(R_i, 4) * J4 * (-(315.0 / 8.0) * (x_i * pow(z_i, 4) / pow(r_i, 11)) + (105.0 / 4.0) * ((x_i * pow(z_i, 2)) / (4 * pow(r_i, 9))) - (15.0 / 8.0) * (x_i / pow(r_i, 7)));
    real grad_y_u_i_shat_j4 = - pow(R_i, 4) * J4 * (-(315.0 / 8.0) * (y_i * pow(z_i, 4) / pow(r_i, 11)) + (105.0 / 4.0) * ((y_i * pow(z_i, 2)) / (4 * pow(r_i, 9))) - (15.0 / 8.0) * (y_i / pow(r_i, 7)));
    real grad_z_u_i_shat_j4 = - pow(R_i, 4) * J4 * ((35.0 / 8.0) * ((4 * pow(z_i, 3) / pow(r_i, 9)) - 9 * pow(z_i, 5) / pow(r_i, 11)) - (15.0 / 4.0) * ((2 * z_i / pow(r_i, 7)) - (7 * pow(z_i, 3) / pow(r_i, 9))) - (15.0 / 8.0) * (z_i / pow(r_i, 7)));

    real grad_x_u_i_shat_j6 = - pow(R_i, 6) * J6 * (-(3003.0 / 16) * (x_i * pow(z_i, 6) / pow(r_i, 15)) + (3465.0 / 16.0) * (x_i * pow(z_i, 4) / pow(r_i, 13)) - (945.0 / 16.0) * (x_i * pow(z_i, 2) / pow(r_i, 11)) + (35.0 / 16.0) * (x_i / pow(r_i,9)));
    real grad_y_u_i_shat_j6 = - pow(R_i, 6) * J6 * (-(3003.0 / 16) * (y_i * pow(z_i, 6) / pow(r_i, 15)) + (3465.0 / 16.0) * (y_i * pow(z_i, 4) / pow(r_i, 13)) - (945.0 / 16.0) * (y_i * pow(z_i, 2) / pow(r_i, 11)) + (35.0 / 16.0) * (y_i / pow(r_i,9)));
    real grad_z_u_i_shat_j6 = - pow(R_i, 6) * J6 * ((231.0 / 16.0) * (6 * pow(z_i, 5) / pow(r_i, 13) - 13 * pow(z_i, 7) / pow(r_i, 15)) - (315.0 / 16.0) * (4 * pow(z_i, 3) / pow(r_i, 11) - 11 * pow(z_i, 5) / pow(r_i, 13)) + (105.0 / 16.0) * (2 * z_i / pow(r_i, 9) - 9 * pow(z_i ,3) / pow(r_i, 11)) + (35.0 / 16.0) * (z_i / pow(r_i, 9)));

    grad_u_i_shat.x = grad_x_u_i_shat_j2 + grad_x_u_i_shat_j3 + grad_x_u_i_shat_j4 + grad_x_u_i_shat_j6;
    grad_u_i_shat.y = grad_y_u_i_shat_j2 + grad_y_u_i_shat_j3 + grad_y_u_i_shat_j4 + grad_y_u_i_shat_j6;
    grad_u_i_shat.z = grad_z_u_i_shat_j2 + grad_z_u_i_shat_j3 + grad_z_u_i_shat_j4 + grad_z_u_i_shat_j6;
}

size_t calculate_j_terms(const real pos[], const real vel[], size_t N, real G, real C, const real masses[], const real radii[], real acc[]) {
    // This function assumes that the 0-th particle is the star, and the 1-th particle is the planet.
    // It calculates the J-terms acting on the satellites of the planets, which starts from i = 2;
    real m_planet = masses[1];
    for (size_t i = 2; i < N; i++) {
        real3 acc_j_terms = {0.0f, 0.0f, 0.0f};
        real3 grad_u_i = {0.0f, 0.0f, 0.0f};
        __calculate_grad_i_shat(pos, radii, i, grad_u_i);
        acc_j_terms.x += (G * (m_planet + masses[i]) * grad_u_i.x);
        acc_j_terms.y += (G * (m_planet + masses[i]) * grad_u_i.y);
        acc_j_terms.z += (G * (m_planet + masses[i]) * grad_u_i.z);

        for (size_t j = 2; j < N; j++) {
            real3 grad_u_j = {0.0, 0.0, 0.0};
            __calculate_grad_i_shat(pos, radii, j, grad_u_j);
            acc_j_terms.x += (G * masses[j] * grad_u_j.x);
            acc_j_terms.y += (G * masses[j] * grad_u_j.y);
            acc_j_terms.z += (G * masses[j] * grad_u_j.z);
        }

        // return the results by adding up the accelerations due to the J terms
        acc[i * 3] += acc_j_terms.x;
        acc[i * 3 + 1] += acc_j_terms.y;
        acc[i * 3 + 2] += acc_j_terms.z;
    }
    return EXIT_NORMAL;
}
