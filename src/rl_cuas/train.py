from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from .env import SwarmDefenseEnv
from .config import TrainConfig


def train_model(cfg: TrainConfig):
    results_dir = Path(cfg.results_dir)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (results_dir / "eval_logs").mkdir(parents=True, exist_ok=True)

    env = Monitor(SwarmDefenseEnv(cfg.scenario, seed=cfg.seed))
    eval_env = Monitor(SwarmDefenseEnv(cfg.scenario, seed=cfg.seed + 123))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.98,
        ent_coef=0.02,
        vf_coef=0.7,
        device="auto",
        tensorboard_log=str(results_dir / "tensorboard_logs"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=str(results_dir / "checkpoints"),
        name_prefix="counter_uav_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(results_dir / "models"),
        log_path=str(results_dir / "eval_logs"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=[checkpoint_cb, eval_cb], progress_bar=True)
    model.save(cfg.model_path)
    return str(Path(cfg.model_path).with_suffix(".zip"))