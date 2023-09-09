import logging
from typing import Any, Callable, List, Tuple, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.collections as collections
import matplotlib.image as image
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray

from utils.metrics import robust_mean, robust_std


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
        log_scale: bool = False,
        colorbar: bool = False,
        projection: str = None,  # type: ignore
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        mpl.rcParams["figure.dpi"] = dpi
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(projection=projection)
        if title:
            ax.set_title(title)

        ax = plotter(ax=ax, color=color, alpha=alpha, **kwargs)

        if legend:
            ax.legend()
        if equal_axes:
            ax.set_aspect("equal", adjustable="box")
        if grid:
            ax.grid()
        if log_scale:
            ax.set_yscale("log")
        if colorbar:
            # Create a separate axes for the colorbar outside of the polar projection
            cax = fig.add_subplot(
                [0.85, 0.1, 0.03, 0.78]
            )  # [left, bottom, width, height]
            axis = ax.get_children()
            if projection == "polar":
                key = collections.QuadMesh
            else:
                key = image.AxesImage

            mesh = _find_instance(axis, key)
            cbar = fig.colorbar(mesh, ax=ax, cax=cax)

        if save:
            if " " in title:
                title = title.replace(" ", "_")
            fig.savefig(path + "/" + title + ".png")

        return fig, ax

    return wrapper


@plot2D
def plot_arms(
    ax: Axes, arms: List[ndarray], color: str = "b", alpha: float = 1, **kwargs
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
                label="robot arm" if idx == 0 else "",
            )

    return ax


@plot2D
def plot_circle(
    ax: Axes,
    origin: ndarray,
    radius: float,
    color: str = "k",
    alpha: float = 1,
    **kwargs,
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
def scatter(
    ax: Axes, data: ndarray, color: str, alpha: float = 1, label: str = "", **kwargs
) -> Axes:
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
def plot_trajectory(
    ax: Axes, trajectory: ndarray, color: str, alpha: float = 1, **kwargs
) -> Axes:
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


@plot2D
def plot_distances(
    ax: Axes,
    distances: ndarray,
    threshold: float = 0,
    color: str = "b",
    alpha: float = 1,
    linestyle: str = "-",
    **kwargs,
) -> Axes:
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
        ax.plot(
            [0, len(distances) - 1],
            [threshold, threshold],
            color="r",
            alpha=alpha,
            label="threshold",
        )

    ax.plot(distances, c=color, alpha=alpha, label="distance", linestyle=linestyle)
    ax.set_xlabel("step")
    ax.set_ylabel("distance")

    return ax


@plot2D
def plot_curves(
    ax: Axes,
    curves: List[ndarray],
    color: str = "b",
    alpha: float = 1,
    label="",
    linestyle: str = "-",
    std: bool = True,
    **kwargs,
) -> Axes:
    """plots a bunch of curves as mean curves and std tunnel

    Args:
        ax (Axes): _description_
        curves (ndarray): a list of curves. Expected curve shape: (num_points, 2). 2 as last dimension for x and y
        alpha (float): _description_ Defaults to 1
        color (str, optional): _description_. Defaults to "b".

    Returns:
        Axes:
    """

    lens = [len(c) for c in curves]
    x = curves[np.argmax(lens)][:, 0]
    mean = robust_mean(curves)[:, 1]

    ax.plot(x, mean, color=color, alpha=alpha, label=label, linestyle=linestyle)

    if std:
        std = robust_std(curves)[:, 1]
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.3 * alpha)

    return ax


@plot2D
def plot_action_distribution(ax: Axes, actions: ndarray, **kwargs) -> ndarray:
    bins = np.linspace(0, 2 * np.pi, 25)
    # bins = None
    num_joints = actions.shape[1]

    if num_joints == 1:
        h, _ = np.histogram(actions, bins=bins)
    elif num_joints == 2:
        h, _, _ = np.histogram2d(actions[:, 0], actions[:, 1], bins=bins)
        # h, x_edges, y_edges= np.histogram2d(actions[:, 0], actions[:, 1])
    else:
        logging.error(f"Not possible to plot action distribution for {num_joints}")
        return ax

    ax.imshow(h, extent=bins[[0, -1, -1, 0]])

    ax.set_xlabel("joint 0")
    ax.set_ylabel("joint 1")

    title = ax.get_title()
    ax.set_title(title + f", n = {len(actions)}")

    return ax


def plot_joint_correlation(ax: Axes, actions: ndarray) -> Axes:
    return ax


@plot2D
def plot_pcolormesh(
    ax: Axes,
    angle: ndarray,
    radius: ndarray,
    z: ndarray,
    alpha: float = 1,
    cmap: str = "viridis",
    stats: bool = False,
    **kwargs,
) -> Axes:
    """plot polar color mesh

    Args:
        ax (Axes): axes to plot on
        angle (ndarray): two dimensions to of angle
        radius (ndarray): two dimensional array of radius
        z (ndarray): two dimensional array of z value
        alpha (float, optional): 1 - transparity value. Defaults to 1.
        cmap (str, optional): colormap. Defaults to "viridis".
        stats (bool): if statistics will be printed or not. Default set to False

    Returns:
        Axes: _description_
    """
    ax.pcolormesh(angle, radius, z, alpha=alpha, cmap=cmap)

    if stats:
        r = 1.75 * max(radius)
        ax.text(
            2.5,
            r,
            r"$\mu$ : " + f"{np.mean(z):.4f}\n" + r"$\sigma$ : " + f"{np.std(z):.4f}",
        )

    return ax


def _find_instance(array: List[Any], key: Type[object]) -> object:
    """finds an instance of an object in a given array

    Args:
        array (List[Any]): array to iterate through
        key (Type[object]): object to find instance of

    Returns:
        object: object
    """
    for element in array:
        if isinstance(element, key):
            return element
    return None
