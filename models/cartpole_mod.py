import crocoddyl
import numpy as np

class ActionModelCartPole(crocoddyl.ActionModelAbstract):
    """
    Defines the action model for the cart-pole system in Crocoddyl,
    with dynamics that exactly match the provided simulation environment.

    The state is [x, θ, ẋ, θ̇] and the control is [force].
    The dynamics and integration match the OpenAI Gym cart-pole implementation.
    """

    def __init__(self,
                 mass_cart: float,
                 mass_pole: float,
                 length_pole: float,
                 delta_t: float,
                 is_terminal: bool = False):
        """
        Initializes the cart-pole action model.

        Args:
            mass_cart (float): Mass of the cart.
            mass_pole (float): Mass of the pole.
            length_pole (float): Length of the pole.
            delta_t (float): Time step for integration.
            is_terminal (bool): True if this is a terminal model (no control input).
        """
        self.is_terminal = is_terminal
        # For a terminal model, there is no control input
        control_dim = 0 if is_terminal else 1
        # State dimension is 4: [x, theta, x_dot, theta_dot]
        # Residual dimension will be state_dim + control_dim for a simple quadratic cost
        residual_dim = 4 + control_dim
        
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(4), control_dim, residual_dim)

        # --- System Parameters (synchronized with simulation) ---
        self.g = 9.81
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.length_pole = length_pole
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.length_pole
        self.Δt = delta_t
        
        # --- Cost Function Weights ---
        # The target state for upright stabilization is [0, 0, 0, 0]
        self.target_state = np.array([0., 0., 0., 0.]) 
        
        # Weights for the quadratic cost: (x-xref)^T W_x (x-xref) + u^T W_u u
        # Tweak these values to change the controller's behavior
        self.state_weights = np.array([
            1.0,  # Penalty on cart position error (x)
            5.0,  # Penalty on pole angle error (theta)
            0.1,  # Penalty on cart velocity (x_dot)
            0.1   # Penalty on pole angular velocity (theta_dot)
        ])
        self.control_weights = np.array([
            0.001 # Penalty on control effort (force)
        ])


    def calc(self, data, x, u=None):
        """
        Computes the next state and cost.
        This is the core of the action model.
        """
        # If this is the terminal model, control `u` is None and has dimension 0.
        if self.is_terminal:
            force = 0.0
        else:
            # Ensure u is not None for a running model
            if u is None:
                u = np.zeros(self.nu)
            force = u[0]

        # Unpack state variables
        cart_x, pole_theta, cart_x_dot, pole_theta_dot = x[0], x[1], x[2], x[3]
        
        # --- Dynamics Calculation (Copied directly from your CartPole class) ---
        sin_theta = np.sin(pole_theta)
        cos_theta = np.cos(pole_theta)

        temp = (force + self.pole_mass_length * pole_theta_dot**2 * sin_theta) / self.total_mass
        
        theta_acc = (self.g * sin_theta - cos_theta * temp) / \
                    (self.length_pole * (4.0/3.0 - self.mass_pole * cos_theta**2 / self.total_mass))
        
        x_acc = temp - (self.pole_mass_length * theta_acc * cos_theta) / self.total_mass

        # --- Integration (Explicit Euler, to match your simulation) ---
        # Note: This is the correct integration scheme based on your `update` method.
        data.xnext[0] = cart_x + self.Δt * cart_x_dot
        data.xnext[1] = pole_theta + self.Δt * pole_theta_dot
        data.xnext[2] = cart_x_dot + self.Δt * x_acc
        data.xnext[3] = pole_theta_dot + self.Δt * theta_acc
        
        # Normalize theta to the range [-pi, pi] to prevent runaway angles
        data.xnext[1] = ((data.xnext[1] + np.pi) % (2 * np.pi)) - np.pi


        # --- Cost Calculation ---
        # The cost is 0.5 * ||r||^2, where r is the residual vector.
        state_error = x - self.target_state
        
        # Ensure angles are compared correctly (e.g., -pi vs pi is 0 error)
        state_error[1] = ((state_error[1] + np.pi) % (2 * np.pi)) - np.pi

        # The residual vector combines weighted state error and control effort.
        if self.is_terminal:
            # For the terminal cost, there is no control penalty
            data.r = self.state_weights * state_error
        else:
            data.r[:4] = self.state_weights * state_error
            data.r[4:] = self.control_weights * u

        data.cost = 0.5 * np.sum(np.square(data.r))


    def calcDiff(self, data, x, u=None):
        """
        Computes the derivatives of the dynamics and cost.
        MPPI does not require this, so we can leave it empty.
        If you were using an algorithm like DDP, you would need to implement this.
        """
        pass