from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class ZoneConfig:
    x: float
    y: float
    z: float
    radius: float
    zone_type: str
    value: float


@dataclass
class ScenarioConfig:
    n_enemies: int = 10
    n_turrets: int = 4
    weapon_range: float = 52.0
    cooldown_time: int = 10
    max_steps: int = 600
    dt: float = 0.1
    kill_radius: float = 11.0
    max_damage: float = 100.0
    zones: List[ZoneConfig] = field(default_factory=lambda: [
        ZoneConfig(0.0, 0.0, 0.0, 12.0, "red", 10.0),
        ZoneConfig(-25.0, 10.0, 0.0, 10.0, "orange", 5.0),
        ZoneConfig(25.0, 10.0, 0.0, 10.0, "orange", 5.0),
    ])


@dataclass
class TrainConfig:
    seed: int = 42
    algo: str = "ppo"
    total_timesteps: int = 200000
    model_path: str = "results/models/commander_policy"
    results_dir: str = "results"
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sc = data.get("scenario", {}) or {}

    scenario = ScenarioConfig(
        n_enemies=sc.get("n_enemies", 10),
        n_turrets=sc.get("n_turrets", 4),
        weapon_range=sc.get("weapon_range", 52.0),
        cooldown_time=sc.get("cooldown_time", 10),
        max_steps=sc.get("max_steps", 600),
        dt=sc.get("dt", 0.1),
        kill_radius=sc.get("kill_radius", 11.0),
        max_damage=sc.get("max_damage", 100.0),
    )

    return TrainConfig(
        seed=data.get("seed", 42),
        algo=data.get("algo", "ppo"),
        total_timesteps=data.get("total_timesteps", 200000),
        model_path=data.get("model_path", "results/models/commander_policy"),
        results_dir=data.get("results_dir", "results"),
        scenario=scenario,
    )