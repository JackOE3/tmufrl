import os
import subprocess
import time
from typing import List
import psutil
import win32gui
import win32process
import win32com.client


class GameInstanceManager:
    def __init__(self, tmi_port: int):
        self.tm_process_id = None
        self.tmi_port = tmi_port
        self.tm_window_id = None
        self.last_game_reboot_seconds = time.perf_counter()

        if self.tmi_port is not None and not (1 <= int(self.tmi_port) <= 65535):
            raise ValueError(f"Invalid port number: {self.tmi_port}")

        self.tml_path = os.environ.get("TMLOADER_PATH")
        if self.tml_path is None:
            raise RuntimeError("TMLOADER_PATH environment variable must be set. It's the path to TMLoader.exe")

        self.tml_profile_name = os.environ.get("TMLOADER_PROFILE_NAME")
        if self.tml_path is None:
            raise RuntimeError("TMLOADER_PROFILE_NAME environment variable must be set. It's the name you set in the Modloader launcher.")


    def _is_tm_process(self, process: psutil.Process) -> bool:
        try:
            return process.name().startswith("TmForever")
        except psutil.NoSuchProcess:
            return False

    def _get_tm_pids(self) -> List[int]:
        return [process.pid for process in psutil.process_iter() if self._is_tm_process(process)]

    def launch_game(self):
        pids_before_launching = self._get_tm_pids()
        assert self.tml_path is not None
        TMLoader_dir = os.path.dirname(self.tml_path)
        cmd = ["run", "TmForever", self.tml_profile_name]
        if self.tmi_port is not None:
            cmd.append(f"/configstring=set custom_port {self.tmi_port}")

        # Launch game asynchronously with Popen
        process = subprocess.Popen(
            [self.tml_path] + cmd,
            cwd=TMLoader_dir,  # Set working directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if game is running (wait until a new pid is found)
        while True:
            pids_after_launching = self._get_tm_pids()
            pids_diff = set(pids_after_launching) - set(pids_before_launching)
            n_pids = len(pids_diff)
            if n_pids > 0:
                assert n_pids == 1, f"Only 1 pid should correspond to a launched game instance. Found: {pids_diff}"
                break
        self.tm_process_id = list(pids_diff)[0]
        self._get_tm_window_id()

        print(f"Launched Trackmania with process id: {self.tm_process_id}")
        self.last_game_reboot_seconds = time.perf_counter()

    def close_game(self):
        subprocess.run(
            ["taskkill", "/PID", str(self.tm_process_id), "/F"],
            text=True,
            check=True
        )

    def _get_tm_window_id(self):
        def get_hwnds_for_pid(pid):
            def callback(hwnd, hwnds):
                _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

                if found_pid == pid:
                    hwnds.append(hwnd)
                return True

            hwnds = []
            win32gui.EnumWindows(callback, hwnds)
            return hwnds

        while True:
            for hwnd in get_hwnds_for_pid(self.tm_process_id):
                if win32gui.GetWindowText(hwnd).startswith("Track"):
                    self.tm_window_id = hwnd
                    return

    def set_foreground_window(self):
        assert self.tm_window_id is not None, "Attempted to set Window to foreground, but did not have a window handle."
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.SetForegroundWindow(self.tm_window_id)
