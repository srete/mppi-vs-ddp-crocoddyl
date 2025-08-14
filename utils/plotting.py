import matplotlib.pyplot as plt
import numpy as np

def plot_control_sequence(control_sequence: np.ndarray,
                          axes: np.ndarray = None,
                          title: str = "Control Sequence",
                          labels: list = None,
                          colors: list = None,
                          linestyles: list = None,
                          linewidth: float = 2):
    """
    Plots a control sequence with multiple dimensions.
    """
    horizon, nu = control_sequence.shape

    show_plot = False
    if axes is None:
        fig, axes = plt.subplots(nu, 1, figsize=(8, 2 * nu), sharex=True)
        if nu == 1:
            axes = np.array([axes]) # Ensure axes is an array even for single subplot
        show_plot = True
    elif len(axes) != nu:
        raise ValueError(f"Provided 'axes' array must have {nu} elements for {nu} control dimensions, "
                         f"but has {len(axes)}.")

    if labels is None:
        labels = [f"Control Dim {i+1}" for i in range(nu)]
    if colors is None:
        colors = [None] * nu # Let matplotlib choose default colors
    if linestyles is None:
        linestyles = ['-'] * nu

    for dim in range(nu):
        ax = axes[dim]
        ax.plot(range(horizon), control_sequence[:, dim],
                label=labels[dim], color=colors[dim], linestyle=linestyles[dim], linewidth=linewidth)
        # ax.set_ylabel(labels[dim])
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time Step (Horizon)")
    if show_plot:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
        plt.show()
        return fig, axes
    else:
        axes[0].set_title(title) # Set title on the first subplot if axes are provided
        return axes

def plot_state_trajectory(state_trajectory: np.ndarray,
                          axes: np.ndarray = None,
                          title: str = "State Trajectory",
                          labels: list = None,
                          colors: list = None,
                          linestyles: list = None,
                          linewidth: float = 2):
    """
    Plots a state trajectory with multiple dimensions.

    """
    horizon_plus_1, nx = state_trajectory.shape

    show_plot = False
    if axes is None:
        fig, axes = plt.subplots(nx, 1, figsize=(8, 2 * nx), sharex=True)
        if nx == 1:
            axes = np.array([axes]) # Ensure axes is an array even for single subplot
        show_plot = True
    elif len(axes) != nx:
        raise ValueError(f"Provided 'axes' array must have {nx} elements for {nx} state dimensions, "
                         f"but has {len(axes)}.")

    if labels is None:
        labels = [f"State Dim {i+1}" for i in range(nx)]
    if colors is None:
        colors = [None] * nx # Let matplotlib choose default colors
    if linestyles is None:
        linestyles = ['-'] * nx

    for dim in range(nx):
        ax = axes[dim]
        ax.plot(range(horizon_plus_1), state_trajectory[:, dim],
                label=labels[dim], color=colors[dim], linestyle=linestyles[dim], linewidth=linewidth)
        # ax.set_ylabel(labels[dim])
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time Step (Horizon + 1)")
    if show_plot:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        return fig, axes
    else:
        axes[0].set_title(title) # Set title on the first subplot if axes are provided
        return axes
    


def plot_cost(costs: list | np.ndarray, ax: plt.Axes = None, title: str = "Cost Over Iterations",
              xlabel: str = "Iteration", ylabel: str = "Cost Value", color: str = 'blue',
              marker: str = None, linestyle: str = '-', label: str = None): # Added label
    """
    Plots a sequence of cost values over iterations.
    """

    if not isinstance(costs, (list, np.ndarray)) or len(costs) == 0:
        print(f"No cost data available to plot for label '{label or 'Unknown'}'.")
        return None

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6)) # Increased default size for standalone plot
        show_plot = True

    ax.plot(costs, marker=marker, linestyle=linestyle, color=color, label=label) # Pass label to plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if show_plot:
        ax.legend() # Add legend if it's a standalone plot
        plt.show()
    return ax

def plot_convergence(costs: list | np.ndarray, ax: plt.Axes = None, title: str = "Convergence Rate",
                     xlabel: str = "Iteration (Change After)", ylabel: str = "Change in Cost",
                     color: str = 'purple', marker: str = None, linestyle: str = '-', label: str = None): # Added label
    """
    Plots the convergence rate (difference between consecutive cost values).
    """
    if not isinstance(costs, (list, np.ndarray)) or len(costs) < 2:
        print(f"Not enough cost data (at least 2 points needed) to calculate convergence rate for label '{label or 'Unknown'}'.")
        return None

    cost_array = np.array(costs)
    conv_rate = np.diff(cost_array)

    show_plot = False
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 6)) # Increased default size for standalone plot
        show_plot = True

    # The x-axis for diff should correspond to the iteration *after* the change
    ax.plot(np.arange(1, len(cost_array)), conv_rate, marker=marker, linestyle=linestyle, color=color, label=label) # Pass label to plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if show_plot:
        ax.legend() # Add legend if it's a standalone plot
        plt.show()
    return ax