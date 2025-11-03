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
