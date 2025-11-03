from gymnasium.envs.registration import register

register(
    id='Trackmania-v0',
    entry_point='tmufrl.envs.tm_gym_env:TrackmaniaEnv',
    kwargs={
        "manager": None,
    }
)
