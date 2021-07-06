from abie.integrator import Integrator
from abie.ode import ODE
import numpy as np
from abie.events import *

__integrator__ = "GaussRadau15"


class GaussRadau15(Integrator):
    def __init__(
        self, particles=None, buffer=None, CONST_G=4 * np.pi ** 2, CONST_C=0.0, deviceID=-1
    ):
        super(self.__class__, self).__init__(particles=particles, 
                                             buffer=buffer, 
                                             CONST_G=CONST_G, 
                                             CONST_C=CONST_C, 
                                             deviceID=deviceID)
        self.tol = 1.0e-9

    def integrate_ctypes(self, to_time=None):
        ret = 0
        try:
            self.libabie.integrator_gr(self.t, to_time, 1.0)

        except CollisionException as e:
            print(e)
            self.handle_collisions(self.libabie.get_collision_data())
            ret = 1
        except CloseEncounterException as e:
            print(e)
            self.store_close_encounters(self.libabie.get_close_encounter_data())
            ret = 2
        finally:
            pos = self.particles.positions.copy()
            vel = self.particles.velocities.copy()
            self.libabie.get_state(
                pos, vel, self.particles.masses, self.particles.radii
            )
            self.particles.positions = pos
            self.particles.velocities = vel
            self._t = self.libabie.get_model_time()
            self.store_state()
            # self._t = to_time

        return ret

    @staticmethod
    def __radau_spacing():
        nh = 8
        h = np.zeros(nh)
        h[0] = 0
        h[1] = 0.0562625605369221464656522
        h[2] = 0.1802406917368923649875799
        h[3] = 0.3526247171131696373739078
        h[4] = 0.5471536263305553830014486
        h[5] = 0.7342101772154105315232106
        h[6] = 0.8853209468390957680903598
        h[7] = 0.9775206135612875018911745
        return h, nh

    @staticmethod
    def __initial_time_step(y0, dy0, G, masses, nbodies):

        p = 15
        ###########   ESTIMATE INITIAL STEP SIZE
        # Compute scaling
        # sc =  abs(y0)*epsb
        # Evaluate function
        f0 = ODE.ode_n_body_second_order(y0, G, masses)
        d0 = max(abs(y0))
        d1 = max(abs(f0))

        if (d0 < 1e-5) or (d1 < 1e-5):
            dt0 = 1e-6
        else:
            dt0 = 0.01 * (d0 / d1)

        # Perform one Euler step
        y1 = y0 + dt0 * dy0
        dy1 = dy0 + dt0 * f0
        # Call function
        f1 = ODE.ode_n_body_second_order(y1, G, masses)
        d2 = max(abs((f1 - f0))) / dt0

        if max(d1, d2) <= 1e-15:
            dt1 = max([1e-6, dt0 * 1e-3])
        else:
            dt1 = (0.01 / max([d1, d2])) ** (1.0 / (p + 1))

        dt = min([100 * dt0, dt1])
        return dt

    @staticmethod
    def __approx_pos(y1, dy1, F1, h, b, T):
        y = y1 + T * h * (
            dy1
            + T
            * h
            * (
                F1
                + h
                * (
                    b[0, :] / 0.3e1
                    + h
                    * (
                        b[1, :] / 0.6e1
                        + h
                        * (
                            b[2, :] / 0.10e2
                            + h
                            * (
                                b[3, :] / 0.15e2
                                + h
                                * (
                                    b[4, :] / 0.21e2
                                    + h * (b[5, :] / 0.28e2 + h * b[6, :] / 0.36e2)
                                )
                            )
                        )
                    )
                )
            )
            / 0.2e1
        )
        return y

    @staticmethod
    def __approx_vel(dy1, F1, h, b, T):
        dy = dy1 + T * h * (
            F1
            + h
            * (
                b[0, :] / 0.2e1
                + h
                * (
                    b[1, :] / 0.3e1
                    + h
                    * (
                        b[2, :] / 0.4e1
                        + h
                        * (
                            b[3, :] / 0.5e1
                            + h
                            * (
                                b[4, :] / 0.6e1
                                + h * (b[5, :] / 0.7e1 + h * b[6, :] / 0.8e1)
                            )
                        )
                    )
                )
            )
        )
        return dy

    @staticmethod
    def __compute_rs():
        # Computed with Maple in (radau_2.mw) with Digits = 50
        r = np.zeros((8, 8), dtype=np.double)

        r[1, 0] = 17.773808914078000840752659565672904106978971632681
        r[2, 0] = 5.5481367185372165056928216140765061758579336941398
        r[3, 0] = 2.8358760786444386782520104428042437400879003147949
        r[4, 0] = 1.8276402675175978297946077587371204385651628457154
        r[5, 0] = 1.3620078160624694969370006292445650994197371928318
        r[6, 0] = 1.1295338753367899027322861542728593509768148769105
        r[7, 0] = 1.0229963298234867458386119071939636779024159134103

        r[2, 1] = 8.0659386483818866885371256689687154412267416180207
        r[3, 1] = 3.3742499769626352599420358188267460448330087696743
        r[4, 1] = 2.0371118353585847827949159161566554921841792590404
        r[5, 1] = 1.4750402175604115479218482480167404024740127431358
        r[6, 1] = 1.2061876660584456166252036299646227791474203527801
        r[7, 1] = 1.0854721939386423840467243172568913862030118679827

        r[3, 2] = 5.8010015592640614823286778893918880155743979164251
        r[4, 2] = 2.7254422118082262837742722003491334729711450288807
        r[5, 2] = 1.8051535801402512604391147435448679586574414080693
        r[6, 2] = 1.4182782637347391537713783674858328433713640692518
        r[7, 2] = 1.2542646222818777659905422465868249586862369725826

        r[4, 3] = 5.1406241058109342286363199091504437929335189668304
        r[5, 3] = 2.6206449263870350811541816031933074696730227729812
        r[6, 3] = 1.8772424961868100972169920283109658335427446084411
        r[7, 3] = 1.6002665494908162609916716949161150366323259154408

        r[5, 4] = 5.3459768998711075141214909632277898045770336660354
        r[6, 4] = 2.9571160172904557478071040204245556508352776929762
        r[7, 4] = 2.3235983002196942228325345451091668073608955835034

        r[6, 5] = 6.6176620137024244874471284891193925737033291491748
        r[7, 5] = 4.1099757783445590862385761824068782144723082633980

        r[7, 6] = 10.846026190236844684706431007823415424143683137181
        return r

    @staticmethod
    def __compute_cs():
        c = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [-0.562625605369221464656522e-1, 1, 0, 0, 0, 0, 0, 0],
                [
                    0.1014080283006362998648180399549641417413495311078e-1,
                    -0.2365032522738145114532321e0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    -0.35758977292516175949344589284567187362040464593728e-2,
                    0.9353769525946206589574845561035371499343547051116e-1,
                    -0.5891279693869841488271399e0,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0.19565654099472210769005672379668610648179838140913e-2,
                    -0.54755386889068686440808430671055022602028382584495e-1,
                    0.41588120008230686168862193041156933067050816537030e0,
                    -0.11362815957175395318285885e1,
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    -0.14365302363708915424459554194153247134438571962198e-2,
                    0.42158527721268707707297347813203202980228135395858e-1,
                    -0.36009959650205681228976647408968845289781580280782e0,
                    0.12501507118406910258505441186857527694077565516084e1,
                    -0.18704917729329500633517991e1,
                    1,
                    0,
                    0,
                ],
                [
                    0.12717903090268677492943117622964220889484666147501e-2,
                    -0.38760357915906770369904626849901899108502158354383e-1,
                    0.36096224345284598322533983078129066420907893718190e0,
                    -0.14668842084004269643701553461378480148761655599754e1,
                    0.29061362593084293014237914371173946705384212479246e1,
                    -0.27558127197720458314421589e1,
                    1,
                    0,
                ],
                [
                    -0.12432012432012432012432013849038719237133940238163e-2,
                    0.39160839160839160839160841227582657239289159887563e-1,
                    -0.39160839160839160839160841545895262429018228668896e0,
                    0.17948717948717948717948719027866738711862551337629e1,
                    -0.43076923076923076923076925231853900723503338586335e1,
                    0.56000000000000000000000001961129300233768803845526e1,
                    -0.37333333333333333333333334e1,
                    1,
                ],
            ]
        )
        return c

    @staticmethod
    def __compute_gs(g, r, ddys, ih):

        F1 = ddys[0, :]
        F2 = ddys[1, :]
        F3 = ddys[2, :]
        F4 = ddys[3, :]
        F5 = ddys[4, :]
        F6 = ddys[5, :]
        F7 = ddys[6, :]
        F8 = ddys[7, :]

        # Update g's with accelerations
        if ih == 1:
            g[0, :] = (F2 - F1) * r[1, 0]
        elif ih == 2:
            g[0, :] = (F2 - F1) * r[1, 0]
            g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
        elif ih == 3:
            g[0, :] = (F2 - F1) * r[1, 0]
            g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
            g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
        elif ih == 4:
            g[0, :] = (F2 - F1) * r[1, 0]
            g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
            g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
            g[3, :] = (
                (((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2]
                - g[2, :]
            ) * r[4, 3]
        elif ih == 5:
            g[0, :] = (F2 - F1) * r[1, 0]
            g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
            g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
            g[3, :] = (
                (((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2]
                - g[2, :]
            ) * r[4, 3]
            g[4, :] = (
                (
                    (((F6 - F1) * r[5, 0] - g[0, :]) * r[5, 1] - g[1, :]) * r[5, 2]
                    - g[2, :]
                )
                * r[5, 3]
                - g[3, :]
            ) * r[5, 4]
        elif ih == 6:
            g[0, :] = (F2 - F1) * r[1, 0]
            g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
            g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
            g[3, :] = (
                (((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2]
                - g[2, :]
            ) * r[4, 3]
            g[4, :] = (
                (
                    (((F6 - F1) * r[5, 0] - g[0, :]) * r[5, 1] - g[1, :]) * r[5, 2]
                    - g[2, :]
                )
                * r[5, 3]
                - g[3, :]
            ) * r[5, 4]
            g[5, :] = (
                (
                    (
                        (((F7 - F1) * r[6, 0] - g[0, :]) * r[6, 1] - g[1, :]) * r[6, 2]
                        - g[2, :]
                    )
                    * r[6, 3]
                    - g[3, :]
                )
                * r[6, 4]
                - g[4, :]
            ) * r[6, 5]
        elif ih == 7:
            g[0, :] = (F2 - F1) * r[1, 0]
            g[1, :] = ((F3 - F1) * r[2, 0] - g[0, :]) * r[2, 1]
            g[2, :] = (((F4 - F1) * r[3, 0] - g[0, :]) * r[3, 1] - g[1, :]) * r[3, 2]
            g[3, :] = (
                (((F5 - F1) * r[4, 0] - g[0, :]) * r[4, 1] - g[1, :]) * r[4, 2]
                - g[2, :]
            ) * r[4, 3]
            g[4, :] = (
                (
                    (((F6 - F1) * r[5, 0] - g[0, :]) * r[5, 1] - g[1, :]) * r[5, 2]
                    - g[2, :]
                )
                * r[5, 3]
                - g[3, :]
            ) * r[5, 4]
            g[5, :] = (
                (
                    (
                        (((F7 - F1) * r[6, 0] - g[0, :]) * r[6, 1] - g[1, :]) * r[6, 2]
                        - g[2, :]
                    )
                    * r[6, 3]
                    - g[3, :]
                )
                * r[6, 4]
                - g[4, :]
            ) * r[6, 5]
            g[6, :] = (
                (
                    (
                        (
                            (((F8 - F1) * r[7, 0] - g[0, :]) * r[7, 1] - g[1, :])
                            * r[7, 2]
                            - g[2, :]
                        )
                        * r[7, 3]
                        - g[3, :]
                    )
                    * r[7, 4]
                    - g[4, :]
                )
                * r[7, 5]
                - g[5, :]
            ) * r[7, 6]
        return g

    @staticmethod
    def __compute_bs_from_gs(b, g, ih, c):

        if ih == 1:
            b[0, :] = (
                c[0, 0] * g[0, :]
                + c[1, 0] * g[1, :]
                + c[2, 0] * g[2, :]
                + c[3, 0] * g[3, :]
                + c[4, 0] * g[4, :]
                + c[5, 0] * g[5, :]
                + c[6, 0] * g[6, :]
            )
        elif ih == 2:
            b[0, :] = (
                c[0, 0] * g[0, :]
                + c[1, 0] * g[1, :]
                + c[2, 0] * g[2, :]
                + c[3, 0] * g[3, :]
                + c[4, 0] * g[4, :]
                + c[5, 0] * g[5, :]
                + c[6, 0] * g[6, :]
            )
            b[1, :] = (
                +c[1, 1] * g[1, :]
                + c[2, 1] * g[2, :]
                + c[3, 1] * g[3, :]
                + c[4, 1] * g[4, :]
                + c[5, 1] * g[5, :]
                + c[6, 1] * g[6, :]
            )
        elif ih == 3:
            b[0, :] = (
                c[0, 0] * g[0, :]
                + c[1, 0] * g[1, :]
                + c[2, 0] * g[2, :]
                + c[3, 0] * g[3, :]
                + c[4, 0] * g[4, :]
                + c[5, 0] * g[5, :]
                + c[6, 0] * g[6, :]
            )
            b[1, :] = (
                +c[1, 1] * g[1, :]
                + c[2, 1] * g[2, :]
                + c[3, 1] * g[3, :]
                + c[4, 1] * g[4, :]
                + c[5, 1] * g[5, :]
                + c[6, 1] * g[6, :]
            )
            b[2, :] = (
                +c[2, 2] * g[2, :]
                + c[3, 2] * g[3, :]
                + c[4, 2] * g[4, :]
                + c[5, 2] * g[5, :]
                + c[6, 2] * g[6, :]
            )
        elif ih == 4:
            b[0, :] = (
                c[0, 0] * g[0, :]
                + c[1, 0] * g[1, :]
                + c[2, 0] * g[2, :]
                + c[3, 0] * g[3, :]
                + c[4, 0] * g[4, :]
                + c[5, 0] * g[5, :]
                + c[6, 0] * g[6, :]
            )
            b[1, :] = (
                +c[1, 1] * g[1, :]
                + c[2, 1] * g[2, :]
                + c[3, 1] * g[3, :]
                + c[4, 1] * g[4, :]
                + c[5, 1] * g[5, :]
                + c[6, 1] * g[6, :]
            )
            b[2, :] = (
                +c[2, 2] * g[2, :]
                + c[3, 2] * g[3, :]
                + c[4, 2] * g[4, :]
                + c[5, 2] * g[5, :]
                + c[6, 2] * g[6, :]
            )
            b[3, :] = (
                c[3, 3] * g[3, :]
                + c[4, 3] * g[4, :]
                + c[5, 3] * g[5, :]
                + c[6, 3] * g[6, :]
            )
        elif ih == 5:
            b[0, :] = (
                c[0, 0] * g[0, :]
                + c[1, 0] * g[1, :]
                + c[2, 0] * g[2, :]
                + c[3, 0] * g[3, :]
                + c[4, 0] * g[4, :]
                + c[5, 0] * g[5, :]
                + c[6, 0] * g[6, :]
            )
            b[1, :] = (
                +c[1, 1] * g[1, :]
                + c[2, 1] * g[2, :]
                + c[3, 1] * g[3, :]
                + c[4, 1] * g[4, :]
                + c[5, 1] * g[5, :]
                + c[6, 1] * g[6, :]
            )
            b[2, :] = (
                +c[2, 2] * g[2, :]
                + c[3, 2] * g[3, :]
                + c[4, 2] * g[4, :]
                + c[5, 2] * g[5, :]
                + c[6, 2] * g[6, :]
            )
            b[3, :] = (
                c[3, 3] * g[3, :]
                + c[4, 3] * g[4, :]
                + c[5, 3] * g[5, :]
                + c[6, 3] * g[6, :]
            )
            b[4, :] = c[4, 4] * g[4, :] + c[5, 4] * g[5, :] + c[6, 4] * g[6, :]
        elif ih == 6:
            b[0, :] = (
                c[0, 0] * g[0, :]
                + c[1, 0] * g[1, :]
                + c[2, 0] * g[2, :]
                + c[3, 0] * g[3, :]
                + c[4, 0] * g[4, :]
                + c[5, 0] * g[5, :]
                + c[6, 0] * g[6, :]
            )
            b[1, :] = (
                +c[1, 1] * g[1, :]
                + c[2, 1] * g[2, :]
                + c[3, 1] * g[3, :]
                + c[4, 1] * g[4, :]
                + c[5, 1] * g[5, :]
                + c[6, 1] * g[6, :]
            )
            b[2, :] = (
                +c[2, 2] * g[2, :]
                + c[3, 2] * g[3, :]
                + c[4, 2] * g[4, :]
                + c[5, 2] * g[5, :]
                + c[6, 2] * g[6, :]
            )
            b[3, :] = (
                c[3, 3] * g[3, :]
                + c[4, 3] * g[4, :]
                + c[5, 3] * g[5, :]
                + c[6, 3] * g[6, :]
            )
            b[4, :] = c[4, 4] * g[4, :] + c[5, 4] * g[5, :] + c[6, 4] * g[6, :]
            b[5, :] = c[5, 5] * g[5, :] + c[6, 5] * g[6, :]
        elif ih == 7:
            b[0, :] = (
                c[0, 0] * g[0, :]
                + c[1, 0] * g[1, :]
                + c[2, 0] * g[2, :]
                + c[3, 0] * g[3, :]
                + c[4, 0] * g[4, :]
                + c[5, 0] * g[5, :]
                + c[6, 0] * g[6, :]
            )
            b[1, :] = (
                +c[1, 1] * g[1, :]
                + c[2, 1] * g[2, :]
                + c[3, 1] * g[3, :]
                + c[4, 1] * g[4, :]
                + c[5, 1] * g[5, :]
                + c[6, 1] * g[6, :]
            )
            b[2, :] = (
                +c[2, 2] * g[2, :]
                + c[3, 2] * g[3, :]
                + c[4, 2] * g[4, :]
                + c[5, 2] * g[5, :]
                + c[6, 2] * g[6, :]
            )
            b[3, :] = (
                c[3, 3] * g[3, :]
                + c[4, 3] * g[4, :]
                + c[5, 3] * g[5, :]
                + c[6, 3] * g[6, :]
            )
            b[4, :] = c[4, 4] * g[4, :] + c[5, 4] * g[5, :] + c[6, 4] * g[6, :]
            b[5, :] = c[5, 5] * g[5, :] + c[6, 5] * g[6, :]
            b[6, :] = c[6, 6] * g[6, :]
        return b

    @staticmethod
    def __refine_bs(b, q, E, it):
        if it != 1:
            bd = b - E
        else:
            bd = b * 0

        q2 = q * q
        q3 = q2 * q
        q4 = q2 * q2
        q5 = q2 * q3
        q6 = q3 * q3
        q7 = q2 * q5

        E[0, :] = q * (
            b[6, :] * 7.0
            + b[5, :] * 6.0
            + b[4, :] * 5.0
            + b[3, :] * 4.0
            + b[2, :] * 3.0
            + b[1, :] * 2.0
            + b[0, :]
        )
        E[1, :] = q2 * (
            b[6, :] * 21.0
            + b[5, :] * 15.0
            + b[4, :] * 10.0
            + b[3, :] * 6.0
            + b[2, :] * 3.0
            + b[1, :]
        )
        E[2, :] = q3 * (
            b[6, :] * 35.0 + b[5, :] * 20.0 + b[4, :] * 10.0 + b[3, :] * 4.0 + b[2, :]
        )
        E[3, :] = q4 * (b[6, :] * 35.0 + b[5, :] * 15.0 + b[4, :] * 5.0 + b[3, :])
        E[4, :] = q5 * (b[6, :] * 21.0 + b[5, :] * 6.0 + b[4, :])
        E[5, :] = q6 * (b[6, :] * 7.0 + b[5, :])
        E[6, :] = q7 * b[6, :]

        b = E + bd
        return [b, E]

    def gaus_radau15_step(
        self,
        y0,
        dy0,
        ddy0,
        ddys,
        dt,
        t,
        tf,
        nh,
        hs,
        bs0,
        bs,
        E,
        g,
        r,
        c,
        tol,
        exponent,
        fac,
        imode,
        G,
        masses,
        nbodies,
    ):

        istat = 0
        while True:
            # Variable number of iterations in PC
            for ipc in range(0, 12):
                ddys = ddys * 0
                # Advance along the Radau sequence
                for ih in range(0, nh):
                    # Estimate position and velocity with bs0 and current h
                    y = self.__approx_pos(y0, dy0, ddy0, hs[ih], bs, dt)
                    dy = self.__approx_vel(dy0, ddy0, hs[ih], bs, dt)
                    # Evaluate force function and store
                    ddys[ih, :] = ODE.ode_n_body_second_order(
                        y, self.CONST_G, self._particles.masses
                    )
                    g = self.__compute_gs(g, r, ddys, ih)
                    bs = self.__compute_bs_from_gs(bs, g, ih, c)
                # Estimate convergence of PC
                db6 = bs[-1, :] - bs0[-1, :]
                if max(abs(db6)) / max(abs(ddys[-1, :])) < 1e-16:
                    break
                bs0 = bs

            # Advance the solution:
            y = self.__approx_pos(y0, dy0, ddy0, 1.0, bs, dt)
            dy = self.__approx_vel(dy0, ddy0, 1.0, bs, dt)
            ddy = ODE.ode_n_body_second_order(y, self.CONST_G, self.particles.masses)

            # Estimate relative error
            estim_b6 = max(abs(bs[-1, :])) / max(abs(ddy))
            err = (estim_b6 / tol) ** (exponent)

            # Step-size required for next step:
            dtreq = dt / err

            # Accept the step
            if err <= 1:

                # Report accepted step:
                istat = 1

                # Advance time:
                t = t + dt

                # Update b coefficients:
                bs0 = bs

                # Refine predictor-corrector coefficients for next pass:
                [bs, E] = self.__refine_bs(bs, dtreq / dt, E, imode)

                # Normal refine mode:
                imode = 1

                # Check if tf was reached:
                if t >= tf:
                    istat = 2

            # Step size for next iteration:
            if dtreq / dt > 1.0 / fac:
                dt = dt / fac
            elif dtreq < 1e-12:
                dt = dt * fac
            else:
                dt = dtreq

            # Correct overshooting:
            if t + dt > tf:
                dt = tf - t
            # Return if the step was accepted:
            if istat > 0:
                break

        return y, dy, ddy, t, dt, g, bs, E, bs0, istat, imode

    def integrate_numpy(self, to_time):
        # Some parameters
        epsb = self.tol  # recommended 1e-9
        fac = 0.25

        # For fixed step integration, choose exponent = 0
        # exponent = 0;
        exponent = 1.0 / 7

        y0 = self.particles.positions
        dy0 = self.particles.velocities
        # Dimension of the system
        dim = len(y0)

        # Tolerance Predictor-Corrector
        tolpc = 1e-18

        # Return Radau spacing
        [hs, nh] = self.__radau_spacing()

        ddy0 = ODE.ode_n_body_second_order(y0, self.CONST_G, self._particles.masses)

        # Initial time step
        self.h = self.__initial_time_step(
            y0, dy0, self.CONST_G, self._particles.masses, self.particles.N
        )

        # Initialize
        bs0 = np.zeros((nh - 1, dim))
        bs = np.zeros((nh - 1, dim))
        g = np.zeros((nh - 1, dim))
        self._t = self.t_start
        E = np.zeros((nh - 1, dim))
        ddys = np.zeros((nh, dim))

        r = self.__compute_rs()
        c = self.__compute_cs()

        integrate = True
        if to_time is not None:
            self.t_end = to_time
        imode = 0
        energy_init = self.calculate_energy()
        while integrate:
            # if self.t + self.h > self.t_end:
            #     self.h = self.t_end - self.t
            #     integrate = False

            # Advance one step and return:
            (
                y,
                dy,
                ddy,
                self._t,
                dt,
                g,
                bs,
                E,
                bs0,
                istat,
                imode,
            ) = self.gaus_radau15_step(
                y0,
                dy0,
                ddy0,
                ddys,
                self.h,
                self.t,
                self.t_end,
                nh,
                hs,
                bs0,
                bs,
                E,
                g,
                r,
                c,
                self.tol,
                exponent,
                fac,
                imode,
                self.CONST_G,
                self._particles.masses,
                self.particles.N,
            )

            # Detect end of integration:
            if istat == 2:
                integrate = False

            # Update step
            y0 = y
            dy0 = dy
            ddy0 = ddy

            self.particles.positions = y
            self.particles.velocities = dy
            self.store_state()
            energy = self.calculate_energy()
            # print('t = %f, E/E0 = %g' % (self.t, np.abs(energy-energy_init)/energy_init))
        self.buf.close()
        return 0
