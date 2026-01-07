from gymnasium.envs.registration import register

# --- Version 0: Basic Physics ---
register(
     id="NeuroShot-v0",
     entry_point="env.neuroshot_v0_env:NeuroShotEnv",
     max_episode_steps=1,
)

# --- Version 0.1: Non-Linear Movement ---
register(
     id="NeuroShot-v0.1",
     entry_point="env.neuroshot_v0_1_env:NeuroShotEnv",
     max_episode_steps=1,
)

# --- Version 0.2: Wind ---
register(
     id="NeuroShot-v0.2",
     entry_point="env.neuroshot_v0_2_env:NeuroShotEnv",
     max_episode_steps=1,
)

# --- Version 1: Wind + Non-Linear Movement ---
register(
     id="NeuroShot-v1",
     entry_point="env.neuroshot_v1_env:NeuroShotEnv",
     max_episode_steps=300,
)