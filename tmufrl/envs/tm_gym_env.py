import socket
import time
import cv2
import numpy as np
from tmufrl.utils.tminterface2 import MessageType, TMInterface, SimStateData
from tmufrl.utils.game_instance_manager import GameInstanceManager
from collections import deque
from typing import Deque, Optional
import gymnasium as gym
from gymnasium import spaces


# Define the Actions dictionary mapping action names to input states
actions_to_inputs_dict = {
    # left, right, accelerate, brake
    "NO_OP": (False, False, False, False),
    "FORWARD": (False, False, True, False),
    "FORWARD_LEFT": (True, False, True, False),
    "FORWARD_RIGHT": (False, True, True, False),
    "LEFT": (True, False, False, False),
    "RIGHT": (False, True, False, False),
    "BRAKE": (False, False, False, True),
    "BRAKE_LEFT": (True, False, False, True),
    "BRAKE_RIGHT": (False, True, False, True),
    "DRIFT_LEFT": (True, False, True, True),
    "DRIFT_RIGHT": (False, True, True, True),
    "FORWARD_BRAKE": (False, False, True, True)
}

# Define the set of actions to be possible in the environment
actions = [
    "NO_OP",
    "FORWARD",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "LEFT",
    "RIGHT",
    "BRAKE",
    "BRAKE_LEFT",
    "BRAKE_RIGHT",
    "DRIFT_LEFT",
    "DRIFT_RIGHT",
    "FORWARD_BRAKE"
]


class TrackmaniaEnv(gym.Env):
    """
    A Gymnasium-like environment wrapper for Trackmania using TMInterface to interact with the game.
    Also keeps a reference to a game manager to handle closing the game when the environment is closed.
    """
    def __init__(self, manager, map_path, max_episode_steps=1000, game_speed=1, game_ticks_per_step=5, image_dim=(120, 160), ):
        super().__init__()
        #print("PORT RECEIVED:", manager.tmi_port)
        if manager is None:
             raise ValueError("The 'manager' (GameInstanceManager) must be provided to TrackmaniaEnv.")
        assert isinstance(manager, GameInstanceManager), "Wrong game manager."
        self.iface = TMInterface(port=manager.tmi_port) # Note: HOST is now defined in tminterface2.py, using that one
        self.manager = manager

        self.timeout_during_runs_ms = 10_000
        self.timeout_between_runs_ms = 100_000
        self.tmi_protection_timeout_s = 500

        self.max_episode_steps = max_episode_steps
        self.game_speed = game_speed

        self.game_ticks_per_step = game_ticks_per_step
        self.GAME_TICK_MS = 10
        self.ms_per_step = self.GAME_TICK_MS * self.game_ticks_per_step

        self.map_path = map_path

        self.current_cp_count = 0
        self.previous_cp_count = 0
        #self.reward_per_m_advanced = 0.01
        # small time penalty per step to incentivize faster driving
        # Linesight: reward_time_penalty_per_s = 1.2
        self.reward_time_penalty_per_s = 0.1
        self.reward_time_penalty_per_step = self.reward_time_penalty_per_s * 1e-3 * self.ms_per_step
        self.reward_for_reaching_checkpoint = 1
        self.reward_for_finishing = 10

        self.last_states: Deque[SimStateData] = deque(maxlen=10)

        self._rewind_requested = False # Flag to indicate if a rewind is requested

        self._frame_requested = False
        self._frame_expected = False

        self.frame_height = image_dim[0]
        self.frame_width = image_dim[1]

        # Gym specific spaces
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.frame_height, self.frame_width, 3), dtype=np.uint8)

        self.current_state = None
        self._previous_state = None
        # Store the very initial state for reliable resets
        self._initial_state = None

        self.has_finished = False
        self.inputs = None

        self.current_step = 0
        self.current_frame = None

        self.UI_disabled = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to the beginning of an episode.

        options: stuff like starting position, map, and so on.

        Returns:
            observation (np.ndarray): The initial observation.2
            info (dict): Auxiliary information.
        """
        super().reset(seed=seed)

        if not self.iface.registered:
            self._connect_and_sync()

        if self._initial_state is None:
            self._start_from_zero()

        self.iface.set_timeout(self.timeout_during_runs_ms)

        # Rewind to the stored initial state
        self._rewind_requested = True

        self._request_next_tmi_run_step()
        self._wait_for_latest_tmi_run_step()

        self.inputs = None
        self.has_finished = False
        self.current_step = 0
        self.current_cp_count = 0
        self.previous_cp_count = 0
        self._previous_dist_to_next_cp = None
        self.last_states.clear()

        assert isinstance(self.current_state, SimStateData)
        assert isinstance(self.current_frame, np.ndarray)

        #obs_sim_state = self._get_observation(self.current_state)
        obs_from_frame = self._get_observation_from_frame(self.current_frame)
        info = self._get_info(self.current_state)

        return obs_from_frame, info

    def step(self, action: int):
        """
        Executes one time step in the environment.

        Args:
            action (int): An action selected from the action space.

        Returns:
            observation (np.ndarray): The agent's observation of the current environment.
            reward (float): The amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended (e.g., race finished).
            truncated (bool): Whether the episode was ended prematurely (e.g., time limit).
            info (dict): Auxiliary information.
        """


        self.current_step += 1
        #print("Starting step", self.iface.port, self.current_step, action)

        # Send action to the game
        self._send_action(action)

        self._request_next_tmi_run_step()
        # Wait for the game (=environment) to execute one step and signal back the current state and frame
        # This is the next state resulting from the action sent above
        self._wait_for_latest_tmi_run_step()

        assert isinstance(self.current_state, SimStateData)
        assert isinstance(self.current_frame, np.ndarray)

        self.last_states.append(self.current_state)

        # Calculate results
        #obs_sim_state = self._get_observation(self.current_state)
        obs_from_frame = self._get_observation_from_frame(self.current_frame)
        reward = self._calculate_reward(self.current_state)
        terminated = self._is_terminated(self.current_state)
        truncated = self._is_truncated(self.current_state)
        info = self._get_info(self.current_state)

        if terminated or truncated:
            self.iface.set_timeout(self.timeout_between_runs_ms)

        #print("Ending step", self.current_step, self.current_state.race_time)
        return obs_from_frame, reward, terminated, truncated, info

    def close(self):
        """Closes the connection to TMInterface."""
        if self.iface and self.iface.registered:
            self.iface.close()
        print(f"[{self.iface.port}] Connection to TMI closed.")

    def _connect_and_sync(self):
        """Connects to the TMInterface server and registers."""
        print(f"[{self.iface.port}] Connecting to TMInterface...")
        while True:
            try:
                # Use the timeout defined for the environment
                self.iface.register(self.tmi_protection_timeout_s)
                print(f"[{self.iface.port}] Registered successfully.")
                # Wait for the initial sync message after registration
                self._wait_for_sync_after_connect()
                break
            except ConnectionRefusedError as e:
                print(f"[{self.iface.port}] Connection refused: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            except socket.timeout:
                print(f"[{self.iface.port}] Registration attempt timed out. Retrying in 5 seconds...")
                time.sleep(5)

    def _wait_for_sync_after_connect(self):
        """Waits for the SC_ON_CONNECT_SYNC message after registering."""
        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                """ self.iface.execute_command(f"set autologin 2")
                self.iface.execute_command("set skip_map_load_screens true")
                self.iface.execute_command("set disable_forced_camera true") """
                self.iface.set_on_step_period(self.GAME_TICK_MS)
                self.iface.execute_command("unload")
                self.iface.set_speed(self.game_speed)
                self._request_map(self.map_path)
                #self.iface.execute_command(f'map "{self.map_path}.Challenge.Gbx"')

                self.iface._respond_to_call(msgtype)
                print(f"[{self.iface.port}] Synced with server.")
                return # Successfully synced
            else:
                # Should ideally not happen right after connect, but handle defensively
                print(f"Warning: Received unexpected message type {msgtype} while waiting for ON_CONNECT_SYNC.")
                # We might need to handle other sync messages if they arrive out of order
                self._handle_server_message(msgtype)

    def _start_from_zero(self):
        """
        From the documentation:

        Note that rewinding to a state in any of these hooks will immediately simulate the next step after the hook. For example, rewinding to a state saved at race time 0, will result in the next call to OnRunStep/OnSimulationStep being at time 10. If you want to apply any immediate input state, make sure to apply it in the same physics step as the call to rewind_to_state.

        That's why I save the initial state at `race_time` = -10
        """

        self.iface.give_up()

        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                if not self.UI_disabled:
                    self.iface.toggle_interface(False)
                    self.UI_disabled = True
                if _time == -10:
                    self._initial_state = self.iface.get_simulation_state()
                    self.iface.set_on_step_period(self.ms_per_step)
                    self.iface._respond_to_call(msgtype)
                    break
                else:
                    self.iface._respond_to_call(msgtype)

    def _send_action(self, action_idx: int):
        """
        Maps the action index to TMInterface inputs, then sents the input state of the car to that input.
        Also responds the the last receives SC_REQUESTED_FRAME_SYNC message, so the game can simulate the next tick using that action.
        """
        action = actions[action_idx]
        inputs_tuple = actions_to_inputs_dict.get(action, (False, False, False, False))
        self.iface.set_input_state(*inputs_tuple)

    def _request_next_tmi_run_step(self):
        """Responds to SC_REQUESTED_FRAME_SYNC, unblocking the server loop so TMI can simulate the next OnRunStep."""
        self.iface._respond_to_call(int(MessageType.SC_REQUESTED_FRAME_SYNC))

    def _wait_for_latest_tmi_run_step(self):
        """
        Waits for SC_RUN_STEP_SYNC and SC_REQUESTED_FRAME_SYNC messages, handling other messages in the meantime.
        Returns the current simulation state and screenshot of the game.
        Does not respond to SC_REQUESTED_FRAME_SYNC, and so effectively pauses the game before simulating the next tick.
        """

        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()

                if self._rewind_requested:
                    self.iface.rewind_to_state(self._initial_state)
                    self._rewind_requested = False

                # Get the new state from the game.
                # If a rewind was requested, this is the state immediately after rewinding.
                self.current_state = self.iface.get_simulation_state()
                _time = self.current_state.race_time

                if _time >= 0 and _time % (self.ms_per_step) == 0:
                    #print("on_run_step", _time)
                    #print(_time, "set speed 0 and requested frame")
                    self.iface.rewind_to_current_state()
                    #iface.set_speed(0) # <- NOT NECESSARY!
                    self.iface.request_frame(self.frame_width, self.frame_height)

                #print(_time, "Responded to run sync")
                self.iface._respond_to_call(msgtype)
            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                # If frame requests are used, handle frame data here
                #print("Received message SC_REQUESTED_FRAME_SYNC")
                self.current_frame = self.iface.get_frame(self.frame_width, self.frame_height)

                #self.iface._respond_to_call(msgtype)
                # Exit loop only when run step is processed and we have the current state and frame
                #print("_wait_for_latest_tmi_run_steps returns")
                return
            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                self.current_cp_count = self.iface._read_int32()
                target_cp_count = self.iface._read_int32()
                #print(f"CP {self.current_cp_count} / {target_cp_count}")

                # Race has finished:
                if self.current_cp_count == target_cp_count:
                    #self.current_state = self.iface.get_simulation_state()
                    #print("got final state", self.current_state.size)
                    self.has_finished = True
                    self.iface.prevent_simulation_finish()
                    self.iface.rewind_to_current_state()
                    self.inputs = self.iface.get_inputs()
                    self.iface.request_frame(self.frame_width, self.frame_height)
                self.iface._respond_to_call(msgtype)
            else:
                self._handle_server_message(msgtype) # Handle other messages

    def _handle_server_message(self, msgtype):
        """Handles incoming server messages and responds. Returns time if it was a run step msg."""

        if msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
            current_lap_count = self.iface._read_int32()
            self.iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.C_SHUTDOWN):
            print("Server requested shutdown.")
            self.close()
            raise ConnectionAbortedError("TMInterface requested shutdown.")

        elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
             # This might happen if connection drops and re-establishes mid-run
             print("Warning: Received SC_ON_CONNECT_SYNC unexpectedly.")
             self.iface._respond_to_call(msgtype)
             # Might need re-initialization logic here

        elif msgtype == int(MessageType.C_RACE_FINISHED):
            print("msgtype race finished")

        else:
            print(f"Warning: Received unhandled message type {msgtype}")
            # Read potentially associated data to clear the buffer if needed (depends on TMI protocol)
            # For now, we assume only known messages requiring response are handled.
            # If TMI sends data with unknown messages, this needs adjustment.
            try:
                 # Try to respond even if unknown, TMI might expect it
                 self.iface._respond_to_call(msgtype)
            except Exception as e:
                 print(f"Could not respond to unknown message {msgtype}: {e}")

    def _get_observation(self, state: SimStateData):
        """Extracts observation features from the simulation state."""
        """ if self.has_finished:
            return None """

        state_dyna_current = state.dyna.current_state
        state_mobil = state.scene_mobil
        state_mobil_engine = state_mobil.engine
        simulation_wheels = state.simulation_wheels

        wheel_state = [simulation_wheels[i].real_time_state for i in range(4)]

        state_position = np.array(
            state_dyna_current.position,
            dtype=np.float32,
        )  # (3,)
        state_velocity = np.array(
            state_dyna_current.linear_speed,
            dtype=np.float32,
        )  # (3,)
        state_angular_speed = np.array(
            state_dyna_current.angular_speed,
            dtype=np.float32,
        )  # (3,)

        velocity_absolute = np.linalg.norm(state_velocity)

        gearbox_state = state_mobil.gearbox_state

        state_rotation_car_to_world = state_dyna_current.rotation.to_numpy()
        state_rotation_world_to_car = state_rotation_car_to_world.T

        if not self.has_finished:
            next_checkpoint = self.checkpoint_centers[self.current_cp_count]
            vec_to_next_cp = next_checkpoint.coord - state_position
            vec_to_next_cp_in_car_reference_frame = state_rotation_world_to_car @ vec_to_next_cp
        else:
            vec_to_next_cp_in_car_reference_frame = np.zeros(3)

        # Positive means CP is in front.
        # Negative means CP is behind.
        cp_forward = vec_to_next_cp_in_car_reference_frame[2]

        # Positive means CP is to the car's left.
        # Negative means CP is to the car's right.
        cp_sideways = vec_to_next_cp_in_car_reference_frame[0]

        # Positive means CP is above the car.
        # Negative means CP is below the car.
        cp_upward = vec_to_next_cp_in_car_reference_frame[1]

        velocity_in_car_reference_frame = state_rotation_world_to_car @ state_velocity

        velocity_fwd = velocity_in_car_reference_frame[2]
        velocity_sideways = velocity_in_car_reference_frame[0]
        velocity_upward = velocity_in_car_reference_frame[1]


        angular_velocity_in_car_reference_frame = state_rotation_world_to_car @ state_angular_speed

        angular_velocity_roll = angular_velocity_in_car_reference_frame[2]
        angular_velocity_pitch = angular_velocity_in_car_reference_frame[0]
        angular_velocity_yaw = angular_velocity_in_car_reference_frame[1]

        # points in the opposide direction of gravity, i.e. up!
        # useless/redundant with yaw/pitch/roll informmation
        negative_gravity_vector_in_car_reference_frame = state_rotation_world_to_car @ np.array([0, 1, 0])

        observation = np.hstack(
            (
                #*(ws.is_sliding for ws in wheel_state),  # Bool
                #*(ws.has_ground_contact for ws in wheel_state),  # Bool
                #*(ws.damper_absorb * 6 for ws in wheel_state),  # 0.005 min, 0.15 max, 0.01 typically -> 6x scaling
                #gearbox_state / 2,  # eg. before gearup this changes temporarily
                #state_mobil_engine.gear / 5,  # 0 -> 5 approx
                #state_mobil_engine.actual_rpm / 1e4,  # 0-10000 approx
                vec_to_next_cp_in_car_reference_frame / (32 * 3), # each coordinate normalized to 3 block lengths
                velocity_absolute * 3.6 / 300, # we normalize to 300 speed
                velocity_in_car_reference_frame * 3.6 / 300, # each component normalized to 300 speed
                angular_velocity_in_car_reference_frame / np.pi, # scale angles by pi
                np.array(state.yaw_pitch_roll)  / np.pi,
            )
        ).astype(np.float32) # shape = (28,)

        return observation

    def _get_observation_from_frame(self, frame_bgra: np.ndarray) -> np.ndarray:
        # https://donadigo.com/tminterface/plugins/api/Graphics/CaptureScreenshot
        # TMInterface will make a screenshot in the BGRA format
        # shape: (H, W, 4)
        assert frame_bgra.shape[-1] == 4, f"Expected BGRA image with 4 channels, got {frame_bgra.shape}"

        # permute and get rid of alpha channel
        frame_rgb = frame_bgra[:, :, [2, 1, 0]]
        return frame_rgb.astype(np.uint8)

    def _calculate_reward(self, state: SimStateData):
        """Calculates the reward based on state changes."""

        reward = 0.0
        cp_just_passed = self.current_cp_count > self.previous_cp_count

        if cp_just_passed:
            if self.has_finished:
                reward += self.reward_for_finishing
            else:
                reward += self.reward_for_reaching_checkpoint
            self.previous_cp_count = self.current_cp_count

        reward -= self.reward_time_penalty_per_step

        return float(reward)

    def _is_terminated(self, state: SimStateData) -> bool:
        """Checks if the episode has terminated (e.g., race finished)."""
        return self.has_finished

    def _is_truncated(self, state: SimStateData):
        """Checks if the episode should be truncated (e.g., time limit or in fail state)."""

        is_stuck = False

        # only check after 2sec of driving
        if state.race_time >= 2000 and len(self.last_states) >= 10:
            is_stuck = True
            max_speed = max(s.display_speed for s in self.last_states)
            if max_speed > 2:
                is_stuck = False

        # check step limit
        reached_step_limit = False
        if self.current_step >= self.max_episode_steps:
             reached_step_limit = True

        return is_stuck or reached_step_limit

    def _get_info(self, state: SimStateData):
        """Returns auxiliary information."""

        info = {
            "race_time": state.player_info.race_time,
            "current_step": self.current_step,
        }

        return info

    def _request_map(self, map_path: str):
        assert self.iface.registered, "TMInterface needs to be registered in order to change the map."
        self.iface.execute_command(f'map "{map_path}.Challenge.Gbx"')

    def __del__(self):
        """Ensure connection is closed when object is deleted."""
        if self.iface and self.iface.registered:
            print("Env object got out of scope and is automatically deleted.")
            self.close()
