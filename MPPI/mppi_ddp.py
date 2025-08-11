import numpy as np
import crocoddyl
from MPPI.mppi import MPPI, MPPILogger

class MPPIDDP(MPPI):
    """
    MPPI optimizer that replaces a specified number of samples with a DDP solution.
    """
    def __init__(self,
                 action_model,
                 terminal_model,
                 horizon: int,
                 num_samples: int,
                 lambda_param: float,
                 noise_sigma: np.ndarray,
                 ddp_problem: crocoddyl.ShootingProblem, # Crocoddyl ShootingProblem for DDP
                 param_gamma: float = 0,
                 param_exploration: float = 0,
                 n_filt: int = 0,
                 logger: MPPILogger = None,
                 num_ddp_replace: int = 1):  # New argument: number of samples to replace with DDP
        
        # Call the base MPPI constructor
        super().__init__(action_model, terminal_model, horizon, num_samples,
                         lambda_param, noise_sigma, param_gamma,
                         param_exploration, n_filt, logger)
        
        self.ddp_problem = ddp_problem
        self.ddp_solver = crocoddyl.SolverDDP(self.ddp_problem)
        self.num_ddp_replace = num_ddp_replace # Store the new argument

        # Ensure num_ddp_replace does not exceed num_samples
        if self.num_ddp_replace > self.num_samples:
            print(f"Warning: num_ddp_replace ({self.num_ddp_replace}) is greater than num_samples ({self.num_samples}). "
                  f"Setting num_ddp_replace to num_samples.")
            self.num_ddp_replace = self.num_samples
        
        # Optional: Add DDP callbacks for verbose output or logging DDP's internal iterations
        # self.ddp_solver.setCallbacks([crocoddyl.CallbackVerbose()])

    def solve(self, x0: np.ndarray, num_iterations: int = 1):
        for iter_idx in range(num_iterations):     
            # This nominal rollout will also provide an initial state trajectory for DDP warm-start
            nominal_X_trajectory, _ = self.rollout_trajectory(x0, self.U_nominal)           
            # --- DDP Solution Step ---
            # Update the DDP problem's initial state for the current iteration
            self.ddp_problem.x0 = x0
            
            # Warm-start DDP with the current nominal controls and nominal state trajectory
            ddp_has_converged = self.ddp_solver.solve(init_us = list(self.U_nominal), init_xs = list(nominal_X_trajectory), maxiter=1) 
            ddp_U = np.array(self.ddp_solver.us)
            ddp_X = np.array(self.ddp_solver.xs)
            ddp_cost = self.ddp_solver.cost

            # Prepare arrays for all samples, including the DDP solution
            costs = np.zeros(self.num_samples)
            curr_iter_sampled_controls = np.zeros((self.num_samples, self.horizon, self.nu))
            curr_iter_sampled_states = np.zeros((self.num_samples, self.horizon + 1, self.nx))
            all_delta_u = np.zeros((self.num_samples, self.horizon, self.nu))

            # Store the DDP solution for the specified number of replacements
            if self.num_ddp_replace > 0:
                costs[:self.num_ddp_replace] = ddp_cost
                curr_iter_sampled_controls[:self.num_ddp_replace] = ddp_U
                curr_iter_sampled_states[:self.num_ddp_replace] = ddp_X
                all_delta_u[:self.num_ddp_replace] = ddp_U - self.U_nominal

            # Generate perturbations for the remaining MPPI samples
            num_mppi_samples = self.num_samples - self.num_ddp_replace
            if num_mppi_samples > 0:
                perturbations = np.random.multivariate_normal(
                    np.zeros(self.nu),
                    self.noise_distribution,
                    size=(num_mppi_samples, self.horizon)
                )
            else:
                perturbations = np.array([]) # Empty array if only DDP samples are used

            # Rollout trajectories and calculate costs for MPPI samples (from index 1 to num_samples-1)
            for k in range(self.num_ddp_replace, self.num_samples):
                # Adjust index for perturbations array (since it has num_mppi_samples elements)
                perturbation_idx = k - self.num_ddp_replace

                # Apply param_exploration logic for MPPI samples
                if perturbation_idx < (1.0 - self.param_exploration) * num_mppi_samples:
                    U_i = self.U_nominal + perturbations[perturbation_idx, :, :]
                else:
                    U_i = perturbations[perturbation_idx, :, :]

                X_trajectory, sample_total_cost = self.rollout_trajectory(x0, U_i, U_nom=self.U_nominal)

                costs[k] = sample_total_cost
                curr_iter_sampled_controls[k] = U_i
                curr_iter_sampled_states[k] = X_trajectory
                all_delta_u[k] = perturbations[perturbation_idx, :, :]

            # Calculate weights based on costs for all samples (including DDP)
            min_cost = np.min(costs)
            weights = np.exp(-1 / self.lambda_param * (costs - min_cost))
            weights /= np.sum(weights)  # Normalize weights

            # Pass the collected sampled data and weights to the logger
            if self.logger and self.logger.enable_logging and (iter_idx % self.logger.n_log == 0):
                self.logger.log_sampled_data(iter_idx,
                                             costs,
                                             curr_iter_sampled_controls,
                                             curr_iter_sampled_states,
                                             weights)
            
            # Calculate the weighted average of control deltas (all_delta_u)
            w_epsilon = np.sum(weights[:, np.newaxis, np.newaxis] * all_delta_u, axis=0)

            # Optional: apply moving average filter
            if self.n_filt != 0:
                w_epsilon = self._moving_average_filter(w_epsilon, window_size=self.n_filt)

            # Update nominal control sequence
            self.U_nominal += w_epsilon
            
            # LOGGING: current total cost and nominal trajectory of the nominal U before update
            if self.logger:
                nominal_X_trajectory, nominal_total_cost = self.rollout_trajectory(x0, self.U_nominal)
                self.logger.log_total_cost(nominal_total_cost)
                self.logger.log_nominal_data(self.U_nominal, nominal_X_trajectory)

            print(f"Iteration {iter_idx + 1}/{num_iterations}, Min Cost: {min_cost:.4f}, Mean Cost: {np.mean(costs):.4f}, DDP Cost: {ddp_cost:.4f}, DDP converged: {ddp_has_converged}")

        return self.U_nominal