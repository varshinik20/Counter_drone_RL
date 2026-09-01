import pandas as pd
from stable_baselines3 import PPO
from .env import SwarmDefenseEnv
from .config import ScenarioConfig
from .policies.baselines import ClassicThreatPolicy, RandomPolicy


def run_episode(policy_name, policy, seed, scenario: ScenarioConfig):
    env = SwarmDefenseEnv(scenario, seed=seed)
    obs, _ = env.reset(seed=seed)
    shots = 0
    tracking_steps = 0
    total_steps = 0
    done = False

    while not done:
        if policy_name == "DeepRL":
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.predict(env)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if action < env.N_ENEMIES:
            shots += 1
        if info["in_range_enemies"] > 0:
            tracking_steps += 1
        total_steps += 1

    util_pct = (shots / max(1.0, (total_steps / env.COOLDOWN_TIME) * env.N_TURRETS)) * 100.0
    track_pct = (tracking_steps / max(1, total_steps)) * 100.0

    return {
        "policy": policy_name,
        "damage_pct": env.total_damage,
        "tracking_pct": track_pct,
        "weapon_utilization_pct": util_pct,
    }


def evaluate_policy(policy_name: str, model_path: str, n_episodes: int, seed: int, scenario: ScenarioConfig):
    if policy_name == "deeprl":
        model = PPO.load(model_path)
        rows = [run_episode("DeepRL", model, seed + i, scenario) for i in range(n_episodes)]
    elif policy_name == "classic":
        policy = ClassicThreatPolicy()
        rows = [run_episode("Classic", policy, seed + i, scenario) for i in range(n_episodes)]
    elif policy_name == "random":
        policy = RandomPolicy()
        rows = [run_episode("Random", policy, seed + i, scenario) for i in range(n_episodes)]
    else:
        raise ValueError("policy must be one of: deeprl, classic, random")

    return pd.DataFrame(rows)