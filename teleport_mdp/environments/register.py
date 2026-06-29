from gymnasium.envs.registration import register, registry

ENV_ID = "teleport_env/FrozenLake-v0"


if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="teleport_mdp.environments.teleport_frozen_lake:TeleportFrozenLakeEnv",
        kwargs={
            "map_name": "4x4",
            "is_slippery": False,
        },
    )
