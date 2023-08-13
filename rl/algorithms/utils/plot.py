import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import ndarray

from matplotlib.figure import Figure

from envs.plane_robot_env.ikrlenv.robots.robot_arm import RobotArm


def scatter_end_points(x_end: ndarray, y_end: ndarray) -> Figure:
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.set_ylim([0, 1])

    radius = np.sqrt(np.power(x_end, 2) + np.power(y_end, 2))
    theta = np.arctan2(y_end, x_end)

    ax.scatter(theta, radius, alpha=0.3)

    return fig


def kde_end_points(
    x_end: ndarray, y_end: ndarray, x_target: ndarray, y_target: ndarray
) -> Figure:
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    end_pos = np.stack([x_end, y_end], axis=1)
    df = pd.DataFrame(end_pos)
    df.columns = ["x", "y"]

    sns.kdeplot(df, x="x", y="y", ax=ax)

    ax.scatter(x_target, y_target, color="r", s=1.5)

    return fig
