import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap module
from matplotlib.colors import Normalize # Import Normalize for colormap

class MPPILogger:
    """
    A dedicated class for logging and visualizing data from the MPPI optimizer.
    """
    def __init__(self, enable_logging: bool = False, n_log: int = 1):
        """
        Initializes the logger.
        """
        self.enable_logging = enable_logging
        self.n_log = n_log

        if self.enable_logging:
            self.total_cost_hist = []  # Logs total cost of the nominal trajectory after each iteration
            # Logs U_nominal at the start of each iteration
            self.nominal_controls_hist = []
            # Logs X_trajectory of the nominal U at the start of each iteration
            self.nominal_states_hist = []

            # Detailed logs (stored only for iterations that are multiples of n_log)
            self.sampled_cost = []
            self.sampled_controls_hist = []  # (num_logged_iters, num_samples, horizon, nu)
            self.sampled_states_hist = []    # (num_logged_iters, num_samples, horizon+1, nx)
            self.weights_hist = []           # (num_logged_iters, num_samples)
            self.iter_indices = [] # To store which actual MPPI iterations have detailed logs

            self.final_U = None # To store the optimized control sequence at the very end
            self.final_X = None
        else:
            self.total_cost_hist = None
            self.nominal_controls_hist = None
            self.nominal_states_hist = None
            self.sampled_controls_hist = None
            self.sampled_states_hist = None
            self.weights_hist = None
            self.iter_indices = None
            self.final_U = None


    def log_total_cost(self, cost: float):
        """Logs the total cost of the nominal trajectory."""
        if self.enable_logging:
            self.total_cost_hist.append(cost)

    def log_nominal_data(self, nominal_u: np.ndarray, nominal_x_trajectory: np.ndarray):
        """Logs the nominal control sequence and its corresponding state trajectory."""
        if self.enable_logging:
            self.nominal_controls_hist.append(nominal_u.copy())
            self.nominal_states_hist.append(nominal_x_trajectory.copy())

    def log_sampled_data(self, iter_idx: int, sampled_cost: np.ndarray, sampled_controls: np.ndarray, 
                         sampled_states: np.ndarray, weights: np.ndarray):
        """Logs all sampled controls, states, and their corresponding weights for specific iterations."""
        if self.enable_logging and (iter_idx % self.n_log == 0):
            self.sampled_cost.append(sampled_cost)
            self.sampled_controls_hist.append(sampled_controls.copy())
            self.sampled_states_hist.append(sampled_states.copy())
            self.weights_hist.append(weights.copy())
            self.iter_indices.append(iter_idx)

    def set_final_controls(self, U_final: np.ndarray):
        """Stores the final optimized nominal control sequence."""
        if self.enable_logging:
            self.final_U = U_final.copy()

    def set_final_state(self, X_optimal: np.ndarray):
        """Stores the final optimized nominal control sequence."""
        if self.enable_logging:
            self.final_X = X_optimal.copy()

    def plot_total_cost(self, ax: plt.Axes = None, title="MPPI Total Cost Over Iterations"):
        """
        Plots the total cost of the nominal trajectory over iterations.
        """
        if not self.enable_logging or not self.total_cost_hist:
            print("Logging is not enabled or no total cost data available to plot.")
            return None

        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True

        ax.plot(self.total_cost_hist, marker='o', linestyle='-', color='blue')
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Cost (Nominal Trajectory)")
        ax.grid(True)

        if show_plot:
            plt.show()
        return ax

    def plot_sampled_controls(self,
                              iteration_index_in_log: int = 0,
                              num_samples_to_plot: int = 10,
                              plot_nominal_controls_for_this_iter: bool = True,
                              plot_final_nominal_controls: bool = False,
                              color_by_weight: bool = False,
                              fig=None, axes: np.ndarray = None, # Expects an array of axes
                              title_prefix="Sampled Controls"):
        """
        Plots a subset of sampled control trajectories for a given logged iteration.
        """
        if not self.enable_logging or not self.sampled_controls_hist:
            print("Logging is not enabled or no sampled control data available to plot.")
            return None

        if iteration_index_in_log >= len(self.sampled_controls_hist) or iteration_index_in_log < 0:
            print(f"Error: iteration_index_in_log {iteration_index_in_log} out of bounds for "
                  f"detailed logs (size {len(self.sampled_controls_hist)}).")
            return None

        controls_data = self.sampled_controls_hist[iteration_index_in_log]
        actual_mppi_iter_idx = self.iter_indices[iteration_index_in_log]
        num_plots = min(num_samples_to_plot, controls_data.shape[0])
        nu = controls_data.shape[2]
        horizon = controls_data.shape[1]

        show_plot = False
        if axes is None:
            fig, axes = plt.subplots(nu, 1, figsize=(8, 2 * nu), sharex=True)
            if nu == 1:
                axes = np.array([axes]) # Ensure axes is an array even for single subplot
            show_plot = True
        elif len(axes) != nu:
            print(f"Error: Provided 'axes' array must have {nu} elements, but has {len(axes)}.")
            return None

        cmap = cm.viridis
        norm = None
        if color_by_weight:
            weights_for_this_iter = self.weights_hist[iteration_index_in_log]
            norm = Normalize(vmin=np.min(weights_for_this_iter), vmax=np.max(weights_for_this_iter))

        for i in range(num_plots):
            color = 'gray'
            # Label only once for non-weighted plots to avoid too many legend entries
            label = 'Sampled Controls' if i == 0 and not color_by_weight else None
            if color_by_weight:
                color = cmap(norm(weights_for_this_iter[i]))
                label = None # No label for individual samples when coloring by weight

            for dim in range(nu):
                axes[dim].plot(range(horizon), controls_data[i, :, dim], color=color, alpha=0.5, label=label if dim == 0 else None)


        # Plot nominal controls for this specific iteration
        if plot_nominal_controls_for_this_iter and self.nominal_controls_hist:
            if actual_mppi_iter_idx < len(self.nominal_controls_hist):
                nominal_u_iter = self.nominal_controls_hist[actual_mppi_iter_idx]
                for dim in range(nu):
                    axes[dim].plot(range(horizon), nominal_u_iter[:, dim], 'k-', linewidth=2.5, label='Nominal U (This Iteration)' if dim == 0 else "")

        # Plot final nominal controls
        if plot_final_nominal_controls and self.final_U is not None:
            for dim in range(nu):
                axes[dim].plot(range(horizon), self.final_U[:, dim], 'r--', linewidth=2.5, label='Final Optimized U' if dim == 0 else "")

        for dim in range(nu):
            axes[dim].set_ylabel(f'Control Dim {dim+1}')
            axes[dim].grid(True)
            # Only show legend on the top subplot to avoid redundancy
            if dim == 0:
                axes[dim].legend()
        axes[-1].set_xlabel("Time Step (Horizon)")

        if color_by_weight:
            cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            cbar.set_label('Sample Weight')

        if show_plot:
            plt.suptitle(f"{title_prefix} (MPPI Iteration {actual_mppi_iter_idx})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
            plt.show()
            return fig, axes
        else:
            # If axes were passed, assume user manages overall layout and show
            axes[0].set_title(f"{title_prefix} (MPPI Iteration {actual_mppi_iter_idx})")
            return axes


    def plot_sampled_states(self,
                            iteration_index_in_log: int = 0,
                            num_samples_to_plot: int = 10,
                            plot_nominal_states_for_this_iter: bool = True,
                            plot_final_nominal_states: bool = False,
                            color_by_weight: bool = False,
                            fig=None, axes: np.ndarray = None, # Expects an array of axes
                            title_prefix="Sampled States"):
        """
        Plots a subset of sampled state trajectories for a given logged iteration.
        """
        if not self.enable_logging or not self.sampled_states_hist:
            print("Logging is not enabled or no sampled state data available to plot.")
            return None
        if iteration_index_in_log >= len(self.sampled_states_hist) or iteration_index_in_log < 0:
            print(f"Error: iteration_index_in_log {iteration_index_in_log} out of bounds for "
                f"detailed logs (size {len(self.sampled_states_hist)}).")
            return None

        states_data = self.sampled_states_hist[iteration_index_in_log]
        actual_mppi_iter_idx = self.iter_indices[iteration_index_in_log]
        num_plots = min(num_samples_to_plot, states_data.shape[0])
        nx = states_data.shape[2]
        horizon_plus_1 = states_data.shape[1]

        show_plot = False
        if axes is None:
            fig, axes = plt.subplots(nx, 1, figsize=(8, 2 * nx), sharex=True)
            if nx == 1:
                axes = np.array([axes]) # Ensure axes is an array even for single subplot
            show_plot = True
        elif len(axes) != nx:
            print(f"Error: Provided 'axes' array must have {nx} elements, but has {len(axes)}.")
            return None

        cmap = cm.viridis
        norm = None
        if color_by_weight:
            if not self.weights_hist or iteration_index_in_log >= len(self.weights_hist):
                print("Warning: 'color_by_weight' is True but no weights data found for this iteration. Plotting with default color.")
                color_by_weight = False
            else:
                weights_for_this_iter = self.weights_hist[iteration_index_in_log]
                norm = Normalize(vmin=np.min(weights_for_this_iter), vmax=np.max(weights_for_this_iter))

        for i in range(num_plots):
            color = 'gray'
            label = 'Sampled States' if i == 0 and not color_by_weight else None
            if color_by_weight:
                color = cmap(norm(weights_for_this_iter[i]))
                label = None

            for dim in range(nx):
                axes[dim].plot(range(horizon_plus_1), states_data[i, :, dim], color=color, alpha=0.5, label=label if dim == 0 else None)


        # Plot nominal state trajectory for this specific iteration
        if plot_nominal_states_for_this_iter and self.nominal_states_hist:
            if actual_mppi_iter_idx < len(self.nominal_states_hist):
                nominal_x_iter = self.nominal_states_hist[actual_mppi_iter_idx]
                for dim in range(nx):
                    axes[dim].plot(range(horizon_plus_1), nominal_x_iter[:, dim], 'k-', linewidth=2.5, label='Nominal X (This Iteration)' if dim == 0 else "")

        # Plot final nominal states
        if plot_final_nominal_states and self.final_X is not None:
            for dim in range(nx):
                axes[dim].plot(range(horizon_plus_1), self.final_X[:, dim], 'r--', linewidth=2.5, label='Final Optimized X' if dim == 0 else "")


        for dim in range(nx):
            axes[dim].set_ylabel(f'State Dim {dim+1}')
            axes[dim].grid(True)
            if dim == 0:
                axes[dim].legend()
        axes[-1].set_xlabel("Time Step (Horizon + 1)")

        if color_by_weight:
            cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            cbar.set_label('Sample Weight')

        if show_plot:
            plt.suptitle(f"{title_prefix} (MPPI Iteration {actual_mppi_iter_idx})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])


            plt.show()
            return fig, axes
        else:
            axes[0].set_title(f"{title_prefix} (MPPI Iteration {actual_mppi_iter_idx})")
            return axes

    def plot_sampled_cost_histogram(self, iteration_index_in_log: int = 0, bins: int = 20, ax: plt.Axes = None):
        """
        Plots a histogram of the sampled costs for a given logged iteration.
        """
        if not self.enable_logging or not self.sampled_cost:
            print("Logging is not enabled or no sampled cost data available to plot.")
            return None

        if iteration_index_in_log >= len(self.sampled_cost) or iteration_index_in_log < 0:
            print(f"Error: iteration_index_in_log {iteration_index_in_log} out of bounds for "
                  f"detailed logs (size {len(self.sampled_cost)}).")
            return None

        costs_for_this_iter = self.sampled_cost[iteration_index_in_log]
        actual_mppi_iter_idx = self.iter_indices[iteration_index_in_log]

        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True

        ax.hist(costs_for_this_iter, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(f"Histogram of Sampled Costs (MPPI Iteration {actual_mppi_iter_idx})")
        ax.set_xlabel("Sampled Cost")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        if show_plot:
            plt.show()
        return ax

    def plot_weights_histogram(self, iteration_index_in_log: int = 0, bins: int = 20, ax: plt.Axes = None):
        """
        Plots a histogram of the weights for a given logged iteration.
        """
        if not self.enable_logging or not self.weights_hist:
            print("Logging is not enabled or no weights data available to plot.")
            return None

        if iteration_index_in_log >= len(self.weights_hist) or iteration_index_in_log < 0:
            print(f"Error: iteration_index_in_log {iteration_index_in_log} out of bounds for "
                  f"detailed logs (size {len(self.weights_hist)}).")
            return None

        weights_for_this_iter = self.weights_hist[iteration_index_in_log]
        actual_mppi_iter_idx = self.iter_indices[iteration_index_in_log]

        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True

        ax.hist(weights_for_this_iter, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(f"Histogram of Sample Weights (MPPI Iteration {actual_mppi_iter_idx})")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.grid(True)

        if show_plot:
            plt.show()
        return ax


# The MPPI Class, modified to use the logger
class MPPI:

    def __init__(self,
                 action_model,
                 terminal_model,
                 horizon: int,
                 num_samples: int,
                 lambda_param: float,
                 noise_sigma: np.ndarray,
                 param_gamma: float = 0,
                 param_exploration: float = 0,
                 n_filt: int = 0,
                 logger: MPPILogger = None): # New argument: Pass the logger object

        self.action_model = action_model
        self.terminal_model = terminal_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_param = lambda_param
        self.noise_sigma = noise_sigma
        self.param_gamma = param_gamma
        self.param_exploration = param_exploration
        self.n_filt = n_filt

        self._running_model_data = self.action_model.createData()
        self._terminal_model_data = self.terminal_model.createData()
        self.nx = self.action_model.state.nx
        self.nu = self.action_model.nu

        # Nominal control sequence (initialized to zeros)
        self.U_nominal = np.zeros((self.horizon, self.nu))
        self.noise_distribution = self._get_noise_distribution()

        # Logger object
        self.logger = logger


    def _moving_average_filter(self, xx: np.ndarray, window_size: int):
        """
        Apply moving average filter for smoothing input sequence.

        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2))
        return xx_mean

    def _get_noise_distribution(self):
        if self.noise_sigma.ndim == 1:
            # If noise_sigma is a vector of std deviations, assume diagonal covariance
            return np.diag(self.noise_sigma**2)
        elif (self.noise_sigma.ndim == 2) and (self.noise_sigma.shape[0] == self.noise_sigma.shape[1]):
            # If it's a full covariance matrix
            return self.noise_sigma
        else:
            raise ValueError("noise_sigma must be a 1D array of standard deviations or a 2D covariance matrix.")


    def rollout_trajectory(self, x0: np.ndarray, U_sequence: np.ndarray, U_nom: np.ndarray = None):

        X_trajectory = [x0]
        current_x = x0
        total_cost = 0.0
        running_data = self._running_model_data
        term_data = self._terminal_model_data

        for t in range(self.horizon):
            u_t = U_sequence[t, :]
            self.action_model.calc(running_data, current_x, u_t)
            next_x = running_data.xnext
            cost_t = running_data.cost
            # print(cost_t)
            if U_nom is not None:
                u_nom_t = U_nom[t, :]
                cost_t += self.param_gamma * u_nom_t.T @ np.linalg.inv(self.noise_distribution) @ u_t
            X_trajectory.append(next_x.copy())
            total_cost += cost_t
            current_x = next_x

        # Terminal cost: l_term(x_T)
        self.terminal_model.calc(term_data, current_x)
        cost_t = term_data.cost
        total_cost += cost_t

        return np.array(X_trajectory), total_cost

    def solve(self, x0: np.ndarray, num_iterations: int = 1):
        for iter_idx in range(num_iterations):

            # Sample control perturbations
            perturbations = np.random.multivariate_normal(
                np.zeros(self.nu),
                self.noise_distribution,
                size=(self.num_samples, self.horizon)
            )

            # LOGGING: Temporary storage for sampled data for the current iteration for detailed logging
            curr_iter_sampled_controls = []
            curr_iter_sampled_states = []
            
            # Rollout trajectories and calculate costs
            costs = np.zeros(self.num_samples) # Store costs to calculate weights
            for k in range(self.num_samples):
                if k < (1.0 - self.param_exploration) * self.num_samples:
                    U_i = self.U_nominal + perturbations[k, :, :]
                else:
                    U_i = perturbations[k, :, :]

                X_trajectory, sample_total_cost = self.rollout_trajectory(x0, U_i, U_nom=self.U_nominal)

                costs[k] = sample_total_cost

                # Collect sampled controls and states if a logger is present and conditions met
                if self.logger and self.logger.enable_logging and (iter_idx % self.logger.n_log == 0):
                    curr_iter_sampled_controls.append(U_i)
                    curr_iter_sampled_states.append(X_trajectory)


            # Calculate weights based on costs
            min_cost = np.min(costs)
            weights = np.exp(-1 / self.lambda_param * (costs - min_cost))
            weights /= np.sum(weights)  # Normalize weights

            # Pass the collected sampled data and weights to the logger if logging conditions are met
            if self.logger and self.logger.enable_logging and (iter_idx % self.logger.n_log == 0):
                self.logger.log_sampled_data(iter_idx,
                                             costs,
                                             np.array(curr_iter_sampled_controls),
                                             np.array(curr_iter_sampled_states),
                                             weights)

            w_epsilon = np.sum(weights[:, np.newaxis, np.newaxis] * perturbations, axis=0)

            # Optionaly: apply moving avarage filter
            if self.n_filt != 0:
                w_epsilon = self._moving_average_filter(w_epsilon, window_size=self.n_filt)

            # Update nominal control sequence
            self.U_nominal += w_epsilon
            print(f"Iteration {iter_idx + 1}/{num_iterations}, Min Cost: {min_cost:.4f}, Mean Cost: {np.mean(costs):.4f}")
            
            # LOGGING: current total cost and nominal trajectory of the nominal U before update
            if self.logger:
                nominal_X_trajectory, nominal_total_cost = self.rollout_trajectory(x0, self.U_nominal)
                self.logger.log_total_cost(nominal_total_cost)
                self.logger.log_nominal_data(self.U_nominal, nominal_X_trajectory)



        return self.U_nominal