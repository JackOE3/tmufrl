# tmufrl

A **Gymnasium** reinforcement learning environment for **Trackmania United Forever** (TMUF), powered by **TMInterface** and **TMLoader**. This package enables RL agents to interact with a running instance of Trackmania United Forever via socket communication.

The [Linesight](https://github.com/Linesight-RL/linesight/tree/main) project is great and runs like butter if you want to train an agent out of the box with a cracked algorithm, but the code itself is pure spaghetti and almost unreadable unless you are 200h balls deep into it yourself. So if you want to try out a specific RL algorithm (Deep Q-Learning is a good start for Trackmania) yourself, you will be helplessly flailing around trying to adopt the code from that project. So, out of that need, this project is born, and as long as you undestand the Gym API, you will be able to do some delicious Reinforcement Learning. May your agent rise to sentience.

This project is still very minimal. The reward function is extremely simple: +1 reward for crossing a checkpoint, +10 reward for finishing the race, and -0.1 reward per second as a time penalty. Should be made more sophisticated in the future. For example, Linesight uses a reference trajectory which is pre-generated to track progress on a map, which gives the agent a meaningful reward at every time step.

---

## Features

- Full **Gymnasium** compatibility
- Real-time game state via **TMInterface**
- Control multiple game instances using `GameInstanceManager`

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/JackOE3/tmufrl.git
cd tmufrl
```

2. **Install in editable mode** (recommended for development):

```bash
pip install -e .
```

> This allows you to modify the code and have changes reflected immediately.

Alternatively, for a standard install:

```bash
pip install .
```

---

**Requirements**:

- Python >= 3.8
- Trackmania United Forever
- [ModLoader](https://tomashu.dev/software/tmloader/)
- [TMInterface](https://donadigo.com/tminterface/)
- TMI Plugin: `Python_Link.as` (put this inside your `TMInterface\Plugins` folder)

> Warning: Windows-only (due to `pywin32` and game dependencies)

---

## Required Environment Variables

Before you can use the environment, set these two environment variables.

Set the path to your TMLoader executable:
`TMLOADER_PATH=C:/Path/To/TMLoader.exe`

Set the profile name to use with TMLoader:
`TMLOADER_PROFILE_NAME=MyTMProfile`

## Setup (TMInterface)

In `TMInterface/config.txt`, add:

```
set autologin 1
```

> Replace `1` with your desired profile number.

---

## Usage

### 1. Start the Game Instance Manager

```python
from tmufrl import GameInstanceManager

# Specify the port that TMInterface will use for this instance
manager = GameInstanceManager(tmi_port=8477)
```

This will:

- Launch `TMLoader.exe` with the specified profile
- Wait for TMInterface to connect on the given port (for this you will need to activate the Python_Link plugin in-game)

### 2. Create the Gym Environment

```python
from tmufrl import GameInstanceManager

# Specify the path to the map you want the agent to play
env = gym.make("Trackmania-v0", manager=manager, map_path="My Challenges/VeryCoolTrack")
```

### 3. Minimal Agent Loop

```python
# Reset environment
obs, info = env.reset()

done = False
total_reward = 0

while not done:
    # Replace with your policy (e.g., random action)
    action = env.action_space.sample()

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated

    if done:
        print(f"Episode finished! Total reward: {total_reward}")

env.close()
manager.close_game()  # Cleanly shut down the game instance
```

### Bonus: Running Multiple Environments (Vectorized)

Use `gymnasium.vector.AsyncVectorEnv` to run **multiple Trackmania instances in parallel**.
This is optimal for on-policy algorithms like Proximal Policy Optimization (PPO) which make heavy use of vectorized environments. However, for off-policy algorithms (like Deep Q-Learning) this setup is suboptimal because the environment will be paused while the agent trains. For reference, the Linesight project uses a Learner-Worker architecture where the network is being trained while multiple agents are simultaneously driving around and collecting experience. It is an added complexity which breaks the vanilla RL loop laid out here by trying to squeeze out the maximum efficiency of parallel computing.

```python
from functools import partial
import gymnasium as gym
from tmufrl.utils.misc import clear_tm_instances, launch_tm_instances

def make_gym_env_fn(manager):
    env = gym.make("Trackmania-v0", manager=manager, map_path="My Challenges/VeryCoolTrack")
    return env

# important to protect entry here so the subprocesses wont execute this code
if __name__ == '__main__':
    # Launch 2 parallel game instances
    N_ENVS = 2
    managers = launch_tm_instances(N_ENVS)  # auto-assigns ports: 8477, 8478

    env_fns = [partial(make_gym_env_fn, manager) for manager in managers]

    envs = gym.vector.AsyncVectorEnv(env_fns)

    obs, infos = envs.reset()

    # Step all environments
    actions = envs.action_space.sample()  # shape: (N_ENVS,)
    obs, rewards, terms, truncs, infos = envs.step(actions)

    # Close all
    envs.close()
    clear_tm_instances()
```

> Each instance runs on a unique `tmi_port` (8477, 8478, ...).
> Use `AsyncVectorEnv` for non-blocking execution.

---

## Observation & Action Space

### Observation Space

- **Type**: `Box(0.0, 1.0, (120, 160), float32)`
- **Default Shape**: `(120, 160)` — grayscale image (normalized to `[0, 1]`)

### Action Space

- **Type**: `Discrete(12)`
- **Actions**:

| Index | Action          |
| ----- | --------------- |
| 0     | `NO_OP`         |
| 1     | `FORWARD`       |
| 2     | `FORWARD_LEFT`  |
| 3     | `FORWARD_RIGHT` |
| 4     | `LEFT`          |
| 5     | `RIGHT`         |
| 6     | `BRAKE`         |
| 7     | `BRAKE_LEFT`    |
| 8     | `BRAKE_RIGHT`   |
| 9     | `DRIFT_LEFT`    |
| 10    | `DRIFT_RIGHT`   |
| 11    | `FORWARD_BRAKE` |

---

### Customizable Environment Parameters

```python
env = gym.make(
    "Trackmania-v0",
    map_path="My Challenges/VeryCoolTrack"  # TM map to load
    manager=manager,
    max_episode_steps=1000,     # Max steps per episode
    game_speed=1,               # Game speed multiplier
    game_ticks_per_step=5,      # Game ticks per env.step()
    image_dim=(120, 160),       # (height, width) of observation
)
```

> Default parameters are shown here.

---

## Notes

- Only tested on **Windows**
- Ensure **TMInterface** is running in the game
- Multiple environments require different `tmi_port` values
- Use `manager.close_game()` to properly terminate the game process

---

## Credits

- **TMInterface Python client (`tminterface2.py`)**:
  [tminterface2.py](https://github.com/Linesight-RL/linesight/blob/main/trackmania_rl/tmi_interaction/tminterface2.py)
  by [Linesight](https://github.com/Linesight-RL/linesight)
- **TMI Plugin**:
  Original [Python_Link](https://github.com/Linesight-RL/linesight/blob/main/trackmania_rl/tmi_interaction/Python_Link.as)
  by [Linesight](https://github.com/Linesight-RL/linesight)
