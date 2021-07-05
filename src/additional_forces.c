#include "common.h"

size_t calculate_additional_forces(const real pos[], const real vel[], size_t N, real G, real C, const real masses[], const real radii[], real acc[]) {
    // ext_acc: the additional accelerations
    // put the routine of calculating additional accelerations here
    // for (size_t i = 0; i < 3 * N; i++) acc[i] -= 0.0; // replace this line

    if (C > 0.0) {
        calculate_post_newtonian(pos, vel, N, G, C, masses, radii, acc);
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
