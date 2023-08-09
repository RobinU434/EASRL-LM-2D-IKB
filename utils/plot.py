from typing import Any, Callable, List, Tuple
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from numpy import ndarray


def plot2D(plotter: Callable[..., Any]):
    def wrapper(
        title: str = "",
        fig: Figure = None,  # type: ignore
        ax: Axes = None,  # type: ignore
        color: str = "b",
        legend: bool = False,
        dpi: int = 300,
        alpha: float = 1,
        save: bool = False,
        path: str = ".",
        equal_axes: bool = False,
        grid: bool = False,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        mpl.rcParams["figure.dpi"] = dpi
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        if len(title) > 0:
            ax.set_title(title)

        ax = plotter(ax=ax, color=color, alpha=alpha, **kwargs)

        if legend:
            ax.legend()
        if equal_axes:
            ax.set_aspect("equal", adjustable="box")
        if grid:
            ax.grid()
        if save:
            if " " in title:
                title = title.replace(" ", "_")
            fig.savefig(path + "/" + title + ".png")

        return fig, ax

    return wrapper


@plot2D
def plot_arms(
    ax: Axes, arms: List[ndarray], color: str = "b", alpha: float = 1
) -> Axes:
    """plot arms

    Args:
        ax (Axes): axes to draw on
        arms (List[ndarray]): list of arm trajectories. Expected shape for arm trajectories: (num_points, num_joints + 1, 2)
        color (str): color to plot in arms
        alpha (float): alpha value to plot arms

    Returns:
        Axes: axes with drawn arms
    """
    for arm in arms:
        for idx, position_sequence in enumerate(arm):
            ax.plot(
                position_sequence[:, 0],
                position_sequence[:, 1],
                color=color,
                marker=".",
                alpha=alpha,
                label="robot arm" if idx == 0 else ""
            )

    return ax


@plot2D
def plot_circle(
    ax: Axes, origin: ndarray, radius: float, color: str = "k", alpha: float = 1
) -> Axes:
    """draw circle in axes

    Args:
        ax (Axes): _description_
        origin (ndarray): _description_
        radius (float): _description_
        color (str, optional): _description_. Defaults to "k".
        alpha (float, optional): _description_. Defaults to 1.

    Returns:
        Axes: _description_
    """

    circle = Circle(origin.tolist(), radius, alpha=alpha, color=color)
    ax.add_patch(circle)

    return ax


@plot2D
def scatter(ax: Axes, data: ndarray, color: str, alpha: float = 1, label:str = "") -> Axes:
    """_summary_

    Args:
        ax (Axes): _description_
        data (ndarray): Expected shape (num_points, 2)
        color (str): _description_
        alpha (float, optional): _description_. Defaults to 1.

    Returns:
        Axes: _description_
    """
    ax.scatter(*data.T, c=color, alpha=alpha, label=label)
    return ax

@plot2D
def plot_trajectory(ax: Axes, trajectory: ndarray, color: str, alpha: float = 1) -> Axes:
    """plot trajectory on axes

    Args:
        ax (Axes): _description_
        trajectory (ndarray): Expected shape, (num_points, 2), with 2 ~ x, y
        color (str): _description_
        alpha (float, optional): _description_. Defaults to 1.

    Returns:
        Axes: _description_
    """
    ax.plot(*trajectory.T, alpha=alpha, color=color, label="end-effector")
    return ax


@plot2D
def plot_distances(ax: Axes, distances: ndarray, threshold: float = 0, color: str = "b", alpha: float = 1) -> Axes:
    """plot distances

    Args:
        ax (Axes): axes to plot distnaces into 
        distances (ndarray): 1D array with distances
        color (str): _description_
        alpha (float, optional): _description_. Defaults to 1.

    Returns:
        Axes: _description_
    """
    if threshold > 0:
        ax.plot([0, len(distances)-1], [threshold, threshold], color="r", alpha=alpha, label="threshold")

    ax.plot(distances, c=color, alpha=alpha, label="distance")
    ax.set_xlabel("step")
    ax.set_ylabel("distance")
    
    return ax
