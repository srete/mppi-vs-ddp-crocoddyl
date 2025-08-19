import crocoddyl
import numpy as np

class ActionModelCartpole(crocoddyl.ActionModelAbstract):
    """
    Defines the action model for the cart-pole system in Crocoddyl.
    
    This model simulates a cart moving along a horizontal track with a 
    freely rotating pole attached to it. The goal is to control the force 
    applied to the cart to stabilize the pole.
    
    State vector: [y, θ, ẏ, θ̇]
    Control input: [f] (force applied to the cart)
    """
    def __init__(self, is_terminal: bool = False):
        """
        Initializes the cart-pole action model with system parameters and cost weights.
        """
        super().__init__(crocoddyl.StateVector(4), 1, 6) # Changed from 6 to 5  # nu = 1; nr = 6

        self.Δt = 0.01 #5e-2     # 0.02 # 5e-2
        self.m_cart = 1.0  # 1.0
        self.m_pole = 0.1  # 0.01 # 0.1
        self.l_pole = 0.5  # 2.0  # 0.5
        self.grav = 9.81   # 9.81
        self.costWeights = [
            1.0,
            1.0,
            0.1,
            0.001,
            0.001,
            1.0,
        ]  # sin, 1-cos, x, xdot, thdot, f

        # self.costWeights = [
        #     5.0,
        #     10,
        #     0.1,
        #     0.1,
        #     0.001
        # ]  # x, theta, xdot, thdot, f


    def calc(self, data, x, u=None):
        """
        Computes the next state and the cost function.

        Args:
            data (ActionDataCartpole): Data structure to store intermediate results.
            x (numpy.ndarray): State vector [y, θ, ẏ, θ̇].
            u (numpy.ndarray, optional): Control input [f]. Defaults to None.

        Updates:
            - Next state xnext.
            - Cost residuals r.
            - Total cost value.
        """
        if u is not None:
            # Getting the state and control variables
            y, θ, ẏ, θ̇, f = x[0], x[1], x[2], x[3], u[0]
            # Shortname for system parameters
            Δt, m_cart, m_pole, l_pole, grav, w = self.Δt, self.m_cart, self.m_pole, self.l_pole, self.grav, self.costWeights
            sin_θ, cos_θ = np.sin(θ), np.cos(θ)
            # Computing the cartpole dynamics
            data.μ = m_cart + m_pole * sin_θ * sin_θ
            data.ÿ̈ = f / data.μ
            data.ÿ̈ += m_pole * grav * cos_θ * sin_θ / data.μ
            data.ÿ̈ -= m_pole * l_pole * (sin_θ * θ̇ ** 2 / data.μ)
            data.θ̈ = (f / l_pole) * cos_θ / data.μ
            data.θ̈ += ((m_cart + m_pole) * grav / l_pole) * sin_θ / data.μ
            data.θ̈ -= m_pole * cos_θ * sin_θ * θ̇**2 / data.μ
            data.ẏ_next = ẏ + Δt * data.ÿ̈
            data.θ̇_next = θ̇ + Δt * data.θ̈
            data.y_next = y + Δt * data.ẏ_next
            data.θ_next = θ + Δt * data.θ̇_next
            data.xnext[:] = np.array([data.y_next, data.θ_next, data.ẏ_next, data.θ̇_next])
            # Computing the cost residual and value
            data.r[:] = w * np.array([sin_θ, 1.0 - cos_θ, y, ẏ, θ̇, f])
            # print(f"shape of w: {len(w)}")
            # print(f"shape of data.r: {data.r.shape}")
            #data.r[1] = ((data.r[1] + np.pi) % (2 * np.pi)) - np.pi
            #data.r[:] = w * np.array([y, θ, ẏ, θ̇, f])
            data.cost = 0.5 * sum(data.r ** 2)
        else:
            # print("U is None")
            # Getting the state and control variables
            y, θ, ẏ, θ̇ = x[0], x[1], x[2], x[3]
            w = self.costWeights
            sin_θ, cos_θ = np.sin(θ), np.cos(θ)
            data.xnext[:] = x
            # Computing the cost residual and value
            data.r[:] = w * np.array([sin_θ, 1.0 - cos_θ, y, ẏ, θ̇, 0.0])
            # data.r[1] = ((data.r[1] + np.pi) % (2 * np.pi)) - np.pi
            # data.r[:] = w * np.array([y, θ, ẏ, θ̇, 0.0])
            data.cost = 0.5 * sum(data.r ** 2)

    def calcDiff(self, data, x, u=None):
        """
        Computes the derivatives of the dynamics and the cost function.

        Args:
            data (ActionDataCartpole): Data structure to store intermediate results.
            x (numpy.ndarray): State vector [y, θ, ẏ, θ̇].
            u (numpy.ndarray, optional): Control input [f]. Defaults to None.

        Updates:
            - Derivetives of the dynamics.
            - Derivatives of the cost function.
        """
        if u is not None:
            # Getting the state and control variables
            y, θ, ẏ, θ̇, f = x[0], x[1], x[2], x[3], u[0]
            # Shortname for system parameters
            Δt, m_cart, m_pole, l_pole, grav, w = self.Δt, self.m_cart, self.m_pole, self.l_pole, self.grav, self.costWeights
            sin_θ, cos_θ = np.sin(θ), np.cos(θ)
            # Computing the derivative of the cartpole dynamics
            data.dμ_dθ = 2.0 * m_pole * sin_θ * cos_θ
            data.dÿ̈_dy = 0.0
            data.dÿ̈_dθ = m_pole * grav * (cos_θ**2 - sin_θ**2) / data.μ
            data.dÿ̈_dθ -= m_pole * l_pole * cos_θ * θ̇ * θ̇ / data.μ
            data.dÿ̈_dθ -= data.dμ_dθ * data.ÿ̈ / data.μ
            data.dÿ̈_dẏ = 0.0
            data.dÿ̈_dθ̇ = -2.0 * m_pole * l_pole * sin_θ * θ̇ / data.μ
            data.dÿ̈_du = 1.0 / data.μ
            data.dθ̈_dy = 0.0
            data.dθ̈_dθ = -(f / l_pole) * sin_θ / data.μ
            data.dθ̈_dθ += ((m_cart + m_pole) * grav / l_pole) * cos_θ / data.μ
            data.dθ̈_dθ -= m_pole * (cos_θ**2 - sin_θ**2) * θ̇ * θ̇ / data.μ
            data.dθ̈_dθ -= data.dμ_dθ * data.θ̈ / data.μ
            data.dθ̈_dẏ = 0.0
            data.dθ̈_dθ̇ = -2.0 * m_pole * cos_θ * sin_θ * θ̇ / data.μ
            data.dθ̈_du = cos_θ / (l_pole * data.μ)
            data.dẏ_next_dy = Δt * data.dÿ̈_dy
            data.dẏ_next_dθ = Δt * data.dÿ̈_dθ
            data.dẏ_next_dẏ = 1.0 + Δt * data.dÿ̈_dẏ
            data.dẏ_next_dθ̇ = Δt * data.dÿ̈_dθ̇
            data.dẏ_next_du = Δt * data.dÿ̈_du
            data.dθ̇_next_dy = Δt * data.dθ̈_dy 
            data.dθ̇_next_dθ = Δt * data.dθ̈_dθ
            data.dθ̇_next_dẏ = Δt * data.dθ̈_dẏ
            data.dθ̇_next_dθ̇ = 1.0 + Δt * data.dθ̈_dθ̇
            data.dθ̇_next_du = Δt * data.dθ̈_du
            data.dy_next_dy = 1.0 + Δt * data.dẏ_next_dy
            data.dy_next_dθ = Δt * data.dẏ_next_dθ
            data.dy_next_dẏ = Δt * data.dẏ_next_dẏ
            data.dy_next_dθ̇ = Δt * data.dẏ_next_dθ̇
            data.dy_next_du = Δt * data.dẏ_next_du
            data.dθ_next_dy = Δt * data.dθ̇_next_dy
            data.dθ_next_dθ = 1.0 + Δt * data.dθ̇_next_dθ
            data.dθ_next_dẏ = Δt * data.dθ̇_next_dẏ
            data.dθ_next_dθ̇ = Δt * data.dθ̇_next_dθ̇
            data.dθ_next_du = Δt * data.dθ̇_next_du
            # Derivatives of the dynamics
            data.Fx[:, :] = np.array([[data.dy_next_dy, data.dy_next_dθ, data.dy_next_dẏ, data.dy_next_dθ̇],
                                      [data.dθ_next_dy, data.dθ_next_dθ, data.dθ_next_dẏ, data.dθ_next_dθ̇],
                                      [data.dẏ_next_dy, data.dẏ_next_dθ, data.dẏ_next_dẏ, data.dẏ_next_dθ̇],
                                      [data.dθ̇_next_dy, data.dθ̇_next_dθ, data.dθ̇_next_dẏ, data.dθ̇_next_dθ̇]])
            data.Fu[:] = np.array([data.dy_next_du, data.dθ_next_du, data.dẏ_next_du, data.dθ̇_next_du])
            # Computing derivatives of the cost function
            w0_2, w1_2, w2_2, w3_2, w4_2, w5_2 = w[0] * w[0], w[1] * w[1], w[2] * w[2], w[3] * w[3], w[4] * w[4], w[5] * w[5]
            #w0_2, w1_2, w2_2, w3_2, w4_2 = w[0] * w[0], w[1] * w[1], w[2] * w[2], w[3] * w[3], w[4] * w[4]
            data.Lx[0] = w2_2 * y
            data.Lx[1] = w0_2 * sin_θ * cos_θ + w1_2 * (1.0 - cos_θ) * sin_θ
            data.Lx[2] = w3_2 * ẏ
            data.Lx[3] = w4_2 * θ̇
            data.Lu[0] = w5_2 * f
            data.Lxx[0, 0] = w2_2
            data.Lxx[1, 1] = w0_2 * (cos_θ * cos_θ - sin_θ * sin_θ)
            data.Lxx[1, 1] += w1_2 * ((1.0 - cos_θ) * cos_θ + sin_θ * sin_θ)
            data.Lxx[2, 2] = w3_2
            data.Lxx[3, 3] = w4_2
            data.Luu[:] = w5_2
            # data.Lx[0] = w0_2 * y
            # data.Lx[1] = w1_2 * θ
            # data.Lx[2] = w2_2 * ẏ
            # data.Lx[3] = w3_2 * θ̇
            # data.Lu[0] = w4_2 * f
            # data.Lxx[0, 0] = w0_2
            # data.Lxx[1, 1] = w1_2
            # data.Lxx[2, 2] = w2_2
            # data.Lxx[3, 3] = w3_2
            # data.Luu[:] = w4_2
            
        else:
            # Getting the state and control variables
            y, θ, ẏ, θ̇ = x[0], x[1], x[2], x[3]
            w = self.costWeights
            sin_θ, cos_θ = np.sin(θ), np.cos(θ)
            # Computing the derivative of the cartpole dynamics
            for i in range(self.state.ndx):
                data.Fx[i, i] = 1.0
            # Computing derivatives of the cost function
            w0_2, w1_2, w2_2, w3_2, w4_2, w5_2 = w[0] * w[0], w[1] * w[1], w[2] * w[2], w[3] * w[3], w[4] * w[4], w[5] * w[5]
            #w0_2, w1_2, w2_2, w3_2, w4_2 = w[0] * w[0], w[1] * w[1], w[2] * w[2], w[3] * w[3], w[4] * w[4]
            data.Lx[0] = w2_2 * y
            data.Lx[1] = w0_2 * sin_θ * cos_θ + w1_2 * (1.0 - cos_θ) * sin_θ
            data.Lx[2] = w3_2 * ẏ
            data.Lx[3] = w4_2 * θ̇
            data.Lxx[0, 0] = w2_2
            data.Lxx[1, 1] = w0_2 * (cos_θ * cos_θ - sin_θ * sin_θ)
            data.Lxx[1, 1] += w1_2 * ((1.0 - cos_θ) * cos_θ + sin_θ * sin_θ)
            data.Lxx[2, 2] = w3_2
            data.Lxx[3, 3] = w4_2
            # data.Lx[0] = w0_2 * y
            # data.Lx[1] = w1_2 * θ
            # data.Lx[2] = w2_2 * ẏ
            # data.Lx[3] = w3_2 * θ̇
            # data.Lxx[0, 0] = w0_2
            # data.Lxx[1, 1] = w1_2
            # data.Lxx[2, 2] = w2_2
            # data.Lxx[3, 3] = w3_2

    def createData(self):
        """
        Creates the action data structure for the cart-pole model.

        Returns:
            ActionDataCartpole: Data structure to store intermediate computations.
        """
        return ActionDataCartpole(self)


class ActionDataCartpole(crocoddyl.ActionDataAbstract):
    """
    Data structure for storing intermediate computations of the cart-pole dynamics.
    """
    def __init__(self, model):
        """
        Initializes the data structure with default values.

        Args:
            model (ActionModelCartpole): Cart-pole action model.
        """
        super().__init__(model)
        self.μ = 0.0
        self.ÿ̈ = 0.0
        self.θ̈ = 0.0
        self.ẏ_next = 0.0
        self.θ̇_next = 0.0
        self.y_next = 0.0
        self.θ_next = 0.0
        self.dμ_dθ = 0.0
        self.dÿ̈_dy = 0.0
        self.dÿ̈_dθ = 0.0
        self.dÿ̈_dθ = 0.0
        self.dÿ̈_dθ = 0.0
        self.dÿ̈_dẏ = 0.0
        self.dÿ̈_dθ̇ = 0.0
        self.dÿ̈_du = 0.0
        self.dθ̈_dy = 0.0
        self.dθ̈_dθ = 0.0
        self.dθ̈_dθ = 0.0
        self.dθ̈_dθ = 0.0
        self.dθ̈_dθ = 0.0
        self.dθ̈_dẏ = 0.0
        self.dθ̈_dθ̇ = 0.0
        self.dẏ_next_dy = 0.0
        self.dẏ_next_dθ = 0.0
        self.dẏ_next_dẏ = 0.0
        self.dẏ_next_dθ̇ = 0.0
        self.dẏ_next_du = 0.0
        self.dθ̇_next_dy = 0.0
        self.dθ̇_next_dθ = 0.0
        self.dθ̇_next_dẏ = 0.0
        self.dθ̇_next_dθ̇ = 0.0
        self.dθ̇_next_du = 0.0
        self.dy_next_dy = 0.0
        self.dy_next_dθ = 0.0
        self.dy_next_dẏ = 0.0
        self.dy_next_dθ̇ = 0.0
        self.dy_next_du = 0.0
        self.dθ_next_dy = 0.0
        self.dθ_next_dθ = 0.0
        self.dθ_next_dẏ = 0.0
        self.dθ_next_dθ̇ = 0.0
        self.dθ_next_du = 0.0