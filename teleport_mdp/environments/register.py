from gymnasium.envs.registration import register

register(
    id="teleport_env/FrozenLake-v0",
    entry_point="teleport_mdp.environments.teleport_frozen_lake:TeleportFrozenLakeEnv",
    kwargs={
        "map_name": "4x4",
        "is_slippery": False,
    },
)
