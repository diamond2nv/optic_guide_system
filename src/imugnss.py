import numpy as np
import matplotlib.pyplot as plt
import os


class IMUGNSS:
    """IMU-GNSS sensor-fusion on the KITTI dataset. The model is the standard 3D
    kinematics model based on inertial inputs and kinematics equations.
    """

    g = np.array([0, 0, -9.82])
    "gravity vector (m/s^2) :math:`\\mathbf{g}`."

    data_dir = os.path.join(os.path.dirname(__file__), "../../examples/data/")

    f_gps = "KittiGps_converted.txt"
    f_imu = "KittiEquivBiasedImu.txt"

    class STATE:
        """State of the system.

        It represents the state of a moving vehicle with IMU biases.

        .. math::

            \\boldsymbol{\\chi} \in \\mathcal{M} = \\left\\{ \\begin{matrix} 
           \\mathbf{C} \in SO(3),
            \\mathbf{v} \in \\mathbb R^3,
            \\mathbf{p} \in \\mathbb R^3,
            \\mathbf{b}_g \in \\mathbb R^3,
            \\mathbf{b}_a \in \\mathbb R^3
           \\end{matrix} \\right\\}

        :ivar Rot: rotation matrix :math:`\mathbf{C}`.
        :ivar v: velocity vector :math:`\mathbf{v}`.
        :ivar p: position vector :math:`\mathbf{p}`.
        :ivar b_gyro: gyro bias :math:`\mathbf{b}_g`.
        :ivar b_acc: accelerometer bias :math:`\mathbf{b}_a`.
        """

        def __init__(self, Rot, v, p, b_gyro, b_acc):
            self.Rot = Rot
            self.v = v
            self.p = p
            self.b_gyro = b_gyro
            self.b_acc = b_acc

    class INPUT:
        """Input of the propagation model.

        The input is a measurement from an Inertial Measurement Unit (IMU).

        .. math:: 

            \\boldsymbol{\\omega} \in \\mathcal{U} = \\left\\{ \\begin{matrix}
            \\mathbf{u} \in \\mathbb R^3,
            \\mathbf{a}_b \in \\mathbb R^3 
            \\end{matrix} \\right\\}

        :ivar gyro: 3D gyro :math:`\mathbf{u}`.
        :ivar acc: 3D accelerometer (measurement in body frame)
              :math:`\mathbf{a}_b`.
        """

        def __init__(self, gyro, acc):
            self.gyro = gyro
            self.acc = acc

    @classmethod
    def f(cls, state, omega, w, dt):
        """ Propagation function.

        .. math::

          \\mathbf{C}_{n+1}  &= \\mathbf{C}_{n} \\exp\\left(\\left(\\mathbf{u}
          - \mathbf{b}_g + \\mathbf{w}^{(0:3)} \\right) dt\\right)  \\\\
          \\mathbf{v}_{n+1}  &= \\mathbf{v}_{n} + \\mathbf{a}  dt, \\\\
          \\mathbf{p}_{n+1}  &= \\mathbf{p}_{n} + \\mathbf{v}_{n} dt 
          + \mathbf{a} dt^2/2 \\\\
          \\mathbf{b}_{g,n+1}  &= \\mathbf{b}_{g,n} 
          + \\mathbf{w}^{(6:9)}dt \\\\
          \\mathbf{b}_{a,n+1}  &= \\mathbf{b}_{a,n} + 
          \\mathbf{w}^{(9:12)} dt     

        where

        .. math::

            \\mathbf{a}  = \\mathbf{C}_{n} 
            \\left( \\mathbf{a}_b -\mathbf{b}_a 
            + \\mathbf{w}^{(3:6)} \\right) + \\mathbf{g}

        Ramdom-walk noises on biases are not added as the Jacobian w.r.t. these 
        noise are trivial. This spares some computations of the UKF.  

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var w: noise :math:`\\mathbf{w}`.
        :var dt: integration step :math:`dt` (s).
        """
        gyro = omega.gyro - state.b_gyro + w[:3]
        acc = state.Rot.dot(omega.acc - state.b_acc + w[3:6]) + cls.g
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO3.exp(gyro*dt)),
            v=state.v + acc*dt,
            p=state.p + state.v*dt + 1/2*acc*dt**2,
            # noise is not added on biases
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
        )
        return new_state

    @classmethod
    def h(cls, state):
        """ Observation function.

        .. math::

            h\\left(\\boldsymbol{\\chi}\\right)  = \\mathbf{p}

        :var state: state :math:`\\boldsymbol{\\chi}`.
        """
        y = state.p
        return y

    @classmethod
    def phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, 
          \\boldsymbol{\\xi}\\right) = \\left( \\begin{matrix}
            \\mathbf{C} \\exp\\left(\\boldsymbol{\\xi}^{(0:3)}\\right) \\\\
            \\mathbf{v} + \\boldsymbol{\\xi}^{(3:6)} \\\\
            \\mathbf{p} + \\boldsymbol{\\xi}^{(6:9)} \\\\
            \\mathbf{b}_g + \\boldsymbol{\\xi}^{(9:12)} \\\\
            \\mathbf{b}_a + \\boldsymbol{\\xi}^{(12:15)}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)
        \\times \\mathbb R^{15}`.

        Its corresponding inverse operation is :meth:`~ukfm.IMUGNSS.phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        new_state = cls.STATE(
            Rot=SO3.exp(xi[:3]).dot(state.Rot),
            v=state.v + xi[3:6],
            p=state.p + xi[6:9],
            b_gyro=state.b_gyro + xi[9:12],
            b_acc=state.b_acc + xi[12:15]
        )
        return new_state

    @classmethod
    def phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) = \\left( \\begin{matrix}
            \\log\\left(\\mathbf{C} \\mathbf{\\hat{C}}^T \\right)\\\\
            \\mathbf{v} - \\mathbf{\\hat{v}} \\\\
            \\mathbf{p} - \\mathbf{\\hat{p}} \\\\
            \\mathbf{b}_g - \\mathbf{\\hat{b}}_g \\\\
            \\mathbf{b}_a - \\mathbf{\\hat{b}}_a
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)
        \\times \\mathbb R^{15}`.

        Its corresponding retraction is :meth:`~ukfm.IMUGNSS.phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = np.hstack([SO3.log(hat_state.Rot.dot(state.Rot.T)),
                        hat_state.v - state.v,
                        hat_state.p - state.p,
                        hat_state.b_gyro - state.b_gyro,
                        hat_state.b_acc - state.b_acc])
        return xi

    @classmethod
    def up_phi(cls, state, xi):
        """Retraction used for updating state and infering Jacobian.

        The retraction :meth:`~ukfm.IMUGNSS.phi` applied on the position state.
        """
        new_state = cls.STATE(
            Rot=state.Rot,
            v=state.v,
            p=xi + state.p,
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
        )
        return new_state

    @classmethod
    def left_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, 
          \\boldsymbol{\\xi}\\right) = \\left( \\begin{matrix}
            \\mathbf{C} \\mathbf{C}_\\mathbf{T} \\\\
            \\mathbf{v} + \\mathbf{C} \\mathbf{r_1} \\\\
            \\mathbf{p} + \\mathbf{C} \\mathbf{r_2} \\\\
            \\mathbf{b}_g + \\boldsymbol{\\xi}^{(9:12)} \\\\
            \\mathbf{b}_a + \\boldsymbol{\\xi}^{(12:15)}
          \\end{matrix} \\right)

        where

        .. math::
            \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}^{(0:9)}\\right) 
            = \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r_1}  &\\mathbf{r}_2 \\\\
                \\mathbf{0}^T & & \\mathbf{I} 
            \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)
        \\times \\mathbb{R}^6` with left multiplication.

        Its corresponding inverse operation is 
        :meth:`~ukfm.IMUGNSS.left_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        dR = SO3.exp(xi[:3])
        J = SO3.left_jacobian(xi[:3])
        new_state = cls.STATE(
            Rot=state.Rot.dot(dR),
            v=state.Rot.dot(J.dot(xi[3:6])) + state.v,
            p=state.Rot.dot(J.dot(xi[6:9])) + state.p,
            b_gyro=state.b_gyro + xi[9:12],
            b_acc=state.b_acc + xi[12:15]
        )
        return new_state

    @classmethod
    def left_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) = \\left( \\begin{matrix}
            \\log\\left(
            \\boldsymbol{\chi}^{-1} \\boldsymbol{\\hat{\\chi}} 
            \\right) \\\\
            \\mathbf{b}_g - \\mathbf{\\hat{b}}_g \\\\
            \\mathbf{b}_a - \\mathbf{\\hat{b}}_a
           \end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)`
        with left multiplication.

        Its corresponding retraction is :meth:`~ukfm.IMUGNSS.left_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        dR = state.Rot.T.dot(hat_state.Rot)
        phi = SO3.log(dR)
        J = SO3.inv_left_jacobian(phi)
        dv = state.Rot.T.dot(hat_state.v - state.v)
        dp = state.Rot.T.dot(hat_state.p - state.p)
        xi = np.hstack([phi,
                        J.dot(dv),
                        J.dot(dp),
                        hat_state.b_gyro - state.b_gyro,
                        hat_state.b_acc - state.b_acc])
        return xi

    @classmethod
    def left_H_ana(cls, state):
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        return H

    @classmethod
    def right_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) 
          = \\left( \\begin{matrix}
            \\mathbf{C}_\\mathbf{T} \\mathbf{C}  \\\\
            \\mathbf{C}_\\mathbf{T}\\mathbf{v} +  \\mathbf{r_1} \\\\
            \\mathbf{C}_\\mathbf{T}\\mathbf{p} +  \\mathbf{r_2} \\\\
            \\mathbf{b}_g + \\boldsymbol{\\xi}^{(9:12)} \\\\
            \\mathbf{b}_a + \\boldsymbol{\\xi}^{(12:15)}
           \\end{matrix} \\right)

        where

        .. math::
            \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}^{(0:9)}\\right)
             = \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r_1}  &\\mathbf{r}_2 \\\\
                \\mathbf{0}^T & & \\mathbf{I} 
            \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)
        \\times \\mathbb{R}^6` with right multiplication.

        Its corresponding inverse operation is 
        :meth:`~ukfm.IMUGNSS.right_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        dR = SO3.exp(xi[:3])
        J = SO3.left_jacobian(xi[:3])
        new_state = cls.STATE(
            Rot=dR.dot(state.Rot),
            v=dR.dot(state.v) + J.dot(xi[3:6]),
            p=dR.dot(state.p) + J.dot(xi[6:9]),
            b_gyro=state.b_gyro + xi[9:12],
            b_acc=state.b_acc + xi[12:15]
        )
        return new_state

    @classmethod
    def right_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::
        
          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) = \\left( \\begin{matrix}
            \\log\\left( \\boldsymbol{\\hat{\\chi}}^{-1} 
            \\boldsymbol{\\chi} \\right) \\\\
            \\mathbf{b}_g - \\mathbf{\\hat{b}}_g \\\\
            \\mathbf{b}_a - \\mathbf{\\hat{b}}_a
          \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)
        \\times \\mathbb{R}^6` with right multiplication.

        Its corresponding retraction is :meth:`~ukfm.IMUGNSS.right_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        dR = hat_state.Rot.dot(state.Rot.T)
        phi = SO3.log(dR)
        J = SO3.inv_left_jacobian(phi)
        dv = hat_state.v - dR*state.v
        dp = hat_state.p - dR*state.p
        xi = np.hstack([phi,
                        J.dot(dv),
                        J.dot(dp),
                        hat_state.b_gyro - state.b_gyro,
                        hat_state.b_acc - state.b_acc])
        return xi

    @classmethod
    def right_up_phi(cls, state, xi):
        """Retraction used for updating state and infering Jacobian.

        The retraction :meth:`~ukfm.IMUGNSS.right_phi` applied on the position 
        state.
        """
        chi = SE3.exp(xi)
        new_state = cls.STATE(
            Rot=chi[:3, :3].dot(state.Rot),
            v=state.v,
            p=chi[:3, 3] + state.p,
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
        )
        return new_state

    @classmethod
    def load(cls, gps_freq):
        data_gps = np.genfromtxt(os.path.join(
            cls.data_dir, cls.f_gps), delimiter=',', skip_header=1)
        data_imu = np.genfromtxt(os.path.join(
            cls.data_dir, cls.f_imu), delimiter=' ', skip_header=1)
        data_imu = data_imu[120:]
        t = data_imu[:, 0]
        t0 = t[0]
        t = t - t0
        N = t.shape[0]

        omegaX = data_imu[:, 5]
        omegaY = data_imu[:, 6]
        omegaZ = data_imu[:, 7]
        accelX = data_imu[:, 2]
        accelY = data_imu[:, 3]
        accelZ = data_imu[:, 4]

        omegas = []
        for n in range(N):
            omegas.append(cls.INPUT(
                gyro=np.array([omegaX[n], omegaY[n], omegaZ[n]]),
                acc=np.array([accelX[n], accelY[n], accelZ[n]])))
        t_gps = data_gps[:, 0] - t0
        N_gps = t_gps.shape[0]

        # vector to know where GPS measurement happen
        one_hot_ys = np.zeros(N)
        k = 1
        ys = np.zeros((N_gps, 3))
        for n in range(1, N):
            if t_gps[k] <= t[n]:
                ys[k] = data_gps[k, 1:]
                one_hot_ys[n] = 1
                k += 1
            if k >= N_gps:
                break
        return omegas, ys, one_hot_ys, t

    @classmethod
    def plot_results(cls, hat_states, ys):
        N = len(hat_states)
        hat_Rots, hat_vs, hat_ps, hat_b_gyros, hat_b_accs = cls.get_states(
            hat_states)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title="Robot position")
        ax.scatter(ys[:, 0], ys[:, 1], c='red')
        plt.plot(hat_ps[:, 0], hat_ps[:, 1], c='blue')
        ax.legend(['UKF', r'GPS measurements'])
        ax.axis('equal')

    @classmethod
    def get_states(cls, states):
        N = len(states)
        Rots = np.zeros((N, 3, 3))
        vs = np.zeros((N, 3))
        ps = np.zeros((N, 3))
        b_gyros = np.zeros((N, 3))
        b_accs = np.zeros((N, 3))
        for n in range(N):
            Rots[n] = states[n].Rot
            vs[n] = states[n].v
            ps[n] = states[n].p
            b_gyros[n] = states[n].b_gyro
            b_accs[n] = states[n].b_acc
        return Rots, vs, ps, b_gyros, b_accs
    
class SO3:
    """Rotation matrix in :math:`SO(3)`

    .. math::

        SO(3) &= \\left\\{ \\mathbf{C} \\in \\mathbb{R}^{3 \\times 3} 
        ~\\middle|~ \\mathbf{C}\\mathbf{C}^T = \\mathbf{1}, \\det
            \\mathbf{C} = 1 \\right\\} \\\\
        \\mathfrak{so}(3) &= \\left\\{ \\boldsymbol{\\Phi} = 
        \\boldsymbol{\\phi}^\\wedge \\in \\mathbb{R}^{3 \\times 3} 
        ~\\middle|~ \\boldsymbol{\\phi} = \\phi \\mathbf{a} \\in \\mathbb{R}
        ^3, \\phi = \\Vert \\boldsymbol{\\phi} \\Vert \\right\\}

    """

    #  tolerance criterion
    TOL = 1e-8
    Id_3 = np.eye(3)

    @classmethod
    def Ad(cls, Rot):
        """Adjoint matrix of the transformation.

        .. math::

            \\text{Ad}(\\mathbf{C}) = \\mathbf{C}
            \\in \\mathbb{R}^{3 \\times 3}

        """
        return Rot

    @classmethod
    def exp(cls, phi):
        """Exponential map for :math:`SO(3)`, which computes a transformation 
        from a tangent vector:

        .. math::

            \\mathbf{C}(\\boldsymbol{\\phi}) =
            \\exp(\\boldsymbol{\\phi}^\wedge) =
            \\begin{cases}
                \\mathbf{1} + \\boldsymbol{\\phi}^\wedge, 
                & \\text{if } \\phi \\text{ is small} \\\\
                \\cos \\phi \\mathbf{1} +
                (1 - \\cos \\phi) \\mathbf{a}\\mathbf{a}^T +
                \\sin \\phi \\mathbf{a}^\\wedge, & \\text{otherwise}
            \\end{cases}

        This is the inverse operation to :meth:`~ukfm.SO3.log`.
        """
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            Rot = cls.Id_3 + SO3.wedge(phi)
        else:
            axis = phi / angle
            c = np.cos(angle)
            s = np.sin(angle)
            Rot = c * cls.Id_3 + (1-c)*np.outer(axis,
                                                axis) + s * cls.wedge(axis)
        return Rot

    @classmethod
    def inv_left_jacobian(cls, phi):
        """:math:`SO(3)` inverse left Jacobian

        .. math::

            \\mathbf{J}^{-1}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} - \\frac{1}{2} \\boldsymbol{\\phi}^\wedge, &
                    \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} \\mathbf{1} +
                \\left( 1 - \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} 
                \\right) \\mathbf{a}\\mathbf{a}^T -
                \\frac{\\phi}{2} \\mathbf{a}^\\wedge, & 
                \\text{otherwise}
            \\end{cases}

        """
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            J = np.eye(3) - 1/2 * cls.wedge(phi)
        else:
            axis = phi / angle
            half_angle = angle/2
            cot = 1 / np.tan(half_angle)
            J = half_angle * cot * cls.Id_3 + \
                (1 - half_angle * cot) * np.outer(axis, axis) -\
                half_angle * cls.wedge(axis)
        return J

    @classmethod
    def left_jacobian(cls, phi):
        """:math:`SO(3)` left Jacobian.

        .. math::

            \\mathbf{J}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} + \\frac{1}{2} \\boldsymbol{\\phi}^\wedge, &
                    \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\sin \\phi}{\\phi} \\mathbf{1} +
                \\left(1 - \\frac{\\sin \\phi}{\\phi} \\right) 
                \\mathbf{a}\\mathbf{a}^T +
                \\frac{1 - \\cos \\phi}{\\phi} \\mathbf{a}^\\wedge, & 
                \\text{otherwise}
            \\end{cases}

        """
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            J = cls.Id_3 - 1/2 * SO3.wedge(phi)
        else:
            axis = phi / angle
            s = np.sin(angle)
            c = np.cos(angle)
            J = (s / angle) * cls.Id_3 + \
                (1 - s / angle) * np.outer(axis, axis) +\
                ((1 - c) / angle) * cls.wedge(axis)
        return J

    @classmethod
    def log(cls, Rot):
        """Logarithmic map for :math:`SO(3)`, which computes a tangent vector 
        from a transformation:

        .. math::

            \\phi &= \\frac{1}{2} 
            \\left( \\mathrm{Tr}(\\mathbf{C}) - 1 \\right) \\\\
            \\boldsymbol{\\phi}(\\mathbf{C}) &=
            \\ln(\\mathbf{C})^\\vee =
            \\begin{cases}
                \\mathbf{1} - \\boldsymbol{\\phi}^\wedge, 
                & \\text{if } \\phi \\text{ is small} \\\\
                \\left( \\frac{1}{2} \\frac{\\phi}{\\sin \\phi} 
                \\left( \\mathbf{C} - \\mathbf{C}^T \\right) 
                \\right)^\\vee, & \\text{otherwise}
            \\end{cases}

        This is the inverse operation to :meth:`~ukfm.SO3.log`.
        """
        cos_angle = 0.5 * np.trace(Rot) - 0.5
        # Clip np.cos(angle) to its proper domain to avoid NaNs from rounding
        # errors
        cos_angle = np.min([np.max([cos_angle, -1]), 1])
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.linalg.norm(angle) < cls.TOL:
            phi = cls.vee(Rot - cls.Id_3)
        else:
            # Otherwise take the matrix logarithm and return the rotation vector
            phi = cls.vee((0.5 * angle / np.sin(angle)) * (Rot - Rot.T))
        return phi

    @classmethod
    def to_rpy(cls, Rot):
        """Convert a rotation matrix to RPY Euler angles 
        :math:`(\\alpha, \\beta, \\gamma)`."""

        pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if np.linalg.norm(pitch - np.pi/2) < cls.TOL:
            yaw = 0
            roll = np.arctan2(Rot[0, 1], Rot[1, 1])
        elif np.linalg.norm(pitch + np.pi/2.) < 1e-9:
            yaw = 0.
            roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(Rot[1, 0] * sec_pitch, Rot[0, 0] * sec_pitch)
            roll = np.arctan2(Rot[2, 1] * sec_pitch, Rot[2, 2] * sec_pitch)

        rpy = np.array([roll, pitch, yaw])
        return rpy

    @classmethod
    def vee(cls, Phi):
        """:math:`SO(3)` vee operator as defined by 
        :cite:`barfootAssociating2014`.

        .. math::

            \\phi = \\boldsymbol{\\Phi}^\\vee

        This is the inverse operation to :meth:`~ukfm.SO3.wedge`.
        """
        phi = np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])
        return phi

    @classmethod
    def wedge(cls, phi):
        """:math:`SO(3)` wedge operator as defined by 
        :cite:`barfootAssociating2014`.

        .. math::

            \\boldsymbol{\\Phi} =
            \\boldsymbol{\\phi}^\\wedge =
            \\begin{bmatrix}
                0 & -\\phi_3 & \\phi_2 \\\\
                \\phi_3 & 0 & -\\phi_1 \\\\
                -\\phi_2 & \\phi_1 & 0
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SO3.vee`.
        """
        Phi = np.array([[0, -phi[2], phi[1]],
                        [phi[2], 0, -phi[0]],
                        [-phi[1], phi[0], 0]])
        return Phi

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles 
        :math:`(\\alpha, \\beta, \\gamma)`.

        .. math:: 
        
            \\mathbf{C} = \\mathbf{C}_z(\\gamma) \\mathbf{C}_y(\\beta)
            \\mathbf{C}_x(\\alpha)

        """
        return cls.rotz(yaw).dot(cls.roty(pitch).dot(cls.rotx(roll)))

    @classmethod
    def rotx(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis.

        .. math::

            \\mathbf{C}_x(\\phi) = 
            \\begin{bmatrix}
                1 & 0 & 0 \\\\
                0 & \\cos \\phi & -\\sin \\phi \\\\
                0 & \\sin \\phi & \\cos \\phi
            \\end{bmatrix}

        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[1., 0., 0.],
                         [0., c, -s],
                         [0., s,  c]])

    @classmethod
    def roty(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the y-axis.

        .. math::

            \\mathbf{C}_y(\\phi) = 
            \\begin{bmatrix}
                \\cos \\phi & 0 & \\sin \\phi \\\\
                0 & 1 & 0 \\\\
                \\sin \\phi & 0 & \\cos \\phi
            \\end{bmatrix}

        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c,  0., s],
                         [0., 1., 0.],
                         [-s, 0., c]])

    @classmethod
    def rotz(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the z-axis.

        .. math::
        
            \\mathbf{C}_z(\\phi) = 
            \\begin{bmatrix}
                \\cos \\phi & -\\sin \\phi & 0 \\\\
                \\sin \\phi  & \\cos \\phi & 0 \\\\
                0 & 0 & 1
            \\end{bmatrix}

        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c, -s,  0.],
                         [s,  c,  0.],
                         [0., 0., 1.]])
        
class SE3:
    """Homogeneous transformation matrix in :math:`SE(3)`.

    .. math::

        SE(3) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r} \\\\
                        \\mathbf{0}^T & 1
                \\end{bmatrix} \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~ 
                \\mathbf{C} \\in SO(3), \\mathbf{r} \\in \\mathbb{R}^3 
                \\right\\} \\\\
        \\mathfrak{se}(3) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~
         \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\boldsymbol{\\phi} \\\\ \\boldsymbol{\\rho}
            \\end{bmatrix} \\in \\mathbb{R}^6, \\boldsymbol{\\rho} \\in 
            \\mathbb{R}^3, \\boldsymbol{\\phi} \in \\mathbb{R}^3 \\right\\}
    """

    @classmethod
    def exp(cls, xi):
        """Exponential map for :math:`SE(3)`, which computes a transformation 
        from a tangent vector:

        .. math::

            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\boldsymbol{\\phi}^\\wedge) & \\mathbf{J} 
                \\boldsymbol{\\rho}  \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SE3.log`.
        """
        chi = np.eye(4)
        chi[:3, :3] = SO3.exp(xi[:3])
        chi[:3, 3] = SO3.left_jacobian(xi[:3]).dot(xi[3:])
        return chi

    @classmethod
    def inv(cls, chi):
        """Inverse map for :math:`SE(3)`.

        .. math::

            \\mathbf{T}^{-1} =
            \\begin{bmatrix}
                \\mathbf{C}^T  & -\\mathbf{C}^T \\boldsymbol{\\rho} 
                    \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        """
        chi_inv = np.eye(4)
        chi_inv[:3, :3] = chi[:3, :3].T
        chi_inv[:3, 3] = -chi[:3, :3].T.dot(chi[:3, 3])
        return chi_inv

    @classmethod
    def log(cls, chi):
        """Logarithmic map for :math:`SE(3)`, which computes a tangent vector 
        from a transformation:

        .. math::
        
            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
                \\mathbf{J} ^ {-1} \\mathbf{r} \\\\
                \\ln(\\boldsymbol{C}) ^\\vee
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SE3.exp`.
        """
        phi = SO3.log(chi[:3, :3])
        xi = np.hstack([phi, SO3.inv_left_jacobian(phi).dot(chi[:3, 3])])
        return xi
    
    
def main():
    pass