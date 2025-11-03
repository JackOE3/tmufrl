import os
import subprocess
from typing import List
import numpy as np
from .game_instance_manager import GameInstanceManager

def run_ps_cmd(cmd: str):
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', cmd])

def clear_tm_instances():
    os.system("taskkill /F /IM TmForever.exe")

def launch_tm_instances(n_windows, base_tmi_port=8477):
    managers = []

    for i in range(n_windows):
        manager = GameInstanceManager(tmi_port=base_tmi_port + i)
        manager.launch_game()
        managers.append(manager)

    return managers


def distance_point_to_rectangle(point: np.ndarray, rectangle: List[np.ndarray]):
    """
    Calculate the shortest distance from a point to a 2D rectangle in 3D space,
    aligned in the xz-plane or yz-plane.

    Parameters:
    - point: the 3D point coordinates [x, y, z]
    - rectangle: list of 4 points which are the vertices of a rectangle which is aligned with the xz- or yz-plane

    Returns:
    - float: The shortest distance from the point to the rectangle.
    """

    x_min, y_min, z_min = 1e4, 1e4, 1e4
    x_max, y_max, z_max = 0, 0, 0

    px, py, pz = point[0], point[1], point[2]

    for [x, y, z] in rectangle:
        if x < x_min:
            x_min = x
        elif x > x_max:
            x_max = x

        if y < y_min:
            y_min = y
        elif y > y_max:
            y_max = y

        if z < z_min:
            z_min = z
        elif z > z_max:
            z_max = z

    if x_min == x_max:
        # Rectangle in zy-plane, x is constant
        z_closest = np.clip(pz, z_min, z_max)
        y_closest = np.clip(py, y_min, y_max)
        distance = np.linalg.norm(np.array([px - x_min, py - y_closest, pz - z_closest]))
    elif z_min == z_max:
        # Rectangle in xy-plane, z is constant
        x_closest = np.clip(px, x_min, x_max)
        y_closest = np.clip(py, y_min, y_max)
        distance = np.linalg.norm(np.array([px - x_closest, py - y_closest, pz - z_min]))
    else:
        raise RuntimeError("No axis-aliged plane was found for the rectangle.")

    return distance
