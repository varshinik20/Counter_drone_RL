import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .config import ScenarioConfig


class SwarmDefenseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ScenarioConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = config or ScenarioConfig()
        self.rng = np.random.default_rng(seed)

        self.N_ENEMIES = self.cfg.n_enemies
        self.N_TURRETS = self.cfg.n_turrets
        self.WEAPON_RANGE = self.cfg.weapon_range
        self.COOLDOWN_TIME = self.cfg.cooldown_time
        self.MAX_STEPS = self.cfg.max_steps
        self.DT = self.cfg.dt
        self.KILL_RADIUS = self.cfg.kill_radius
        self.MAX_DAMAGE = self.cfg.max_damage

        self.zones = [
            {
                "pos": np.array([z.x, z.y, z.z], dtype=np.float32),
                "rad": z.radius,
                "type": z.zone_type,
                "value": z.value,
            }
            for z in self.cfg.zones
        ]

        self.turrets = [
            {"pos": np.array([-15.0, -5.0, 0.0], dtype=np.float32), "cooldown": 0},
            {"pos": np.array([15.0, -5.0, 0.0], dtype=np.float32), "cooldown": 0},
            {"pos": np.array([-5.0, 5.0, 0.0], dtype=np.float32), "cooldown": 0},
            {"pos": np.array([5.0, 5.0, 0.0], dtype=np.float32), "cooldown": 0},
        ][: self.N_TURRETS]

        self.last_shot = None
        self.last_hit = False
        self.last_impact = None

        self.action_space = spaces.Discrete(self.N_ENEMIES + 1)

        enemy_features = 12
        turret_features = self.N_TURRETS
        global_features = 6
        obs_dim = self.N_ENEMIES * enemy_features + turret_features + global_features
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.total_damage = 0.0
        self.current_step = 0
        self.last_shot = None
        self.last_hit = False
        self.last_impact = None

        for turret in self.turrets:
            turret["cooldown"] = 0

        self.enemies = [self._spawn_enemy(i) for i in range(self.N_ENEMIES)]
        return self._get_obs(), {}

    def _spawn_enemy(self, enemy_id: int):
        x = self.rng.uniform(-40, 40)
        y = self.rng.uniform(50, 60)
        z = self.rng.uniform(10, 30)
        pos = np.array([x, y, z], dtype=np.float32)

        target_zone = int(self.rng.integers(0, len(self.zones)))
        zone = self.zones[target_zone]
        direction = zone["pos"] - pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        speed = self.rng.uniform(1.6, 3.2)
        vel = (direction * speed).astype(np.float32)

        chassis = int(self.rng.integers(1, 4))
        explosive = int(self.rng.integers(1, 4))
        damage_value = float(chassis * explosive * 2.5)

        return {
            "id": enemy_id,
            "pos": pos,
            "vel": vel,
            "active": 1.0,
            "target_zone": target_zone,
            "chassis": chassis,
            "explosive": explosive,
            "damage_value": damage_value,
        }

    def _zone_distance(self, enemy):
        zone = self.zones[enemy["target_zone"]]
        return float(np.linalg.norm(enemy["pos"] - zone["pos"]))

    def _time_to_zone(self, enemy):
        speed = float(np.linalg.norm(enemy["vel"]))
        return self._zone_distance(enemy) / max(speed, 1e-6)

    def _best_turret_for_enemy(self, enemy):
        best_turret = None
        min_dist = float("inf")
        for i, turret in enumerate(self.turrets):
            if turret["cooldown"] > 0:
                continue
            dist = np.linalg.norm(turret["pos"] - enemy["pos"])
            if dist < self.WEAPON_RANGE and dist < min_dist:
                min_dist = dist
                best_turret = i
        return best_turret, min_dist

    def _threat_score(self, enemy):
        zone = self.zones[enemy["target_zone"]]
        dist_zone = self._zone_distance(enemy)
        time_zone = self._time_to_zone(enemy)
        urgency_dist = max(0.0, 60.0 - dist_zone) * 0.25
        urgency_time = max(0.0, 20.0 - time_zone) * 1.2
        return zone["value"] * 5.0 + enemy["damage_value"] * 2.5 + urgency_dist + urgency_time

    def _kill_probability(self, turret_dist):
        # easier than before so PPO learns shooting behavior sooner
        norm_dist = turret_dist / max(self.WEAPON_RANGE, 1e-6)
        pk = 0.95 - 0.45 * norm_dist
        return float(np.clip(pk, 0.35, 0.95))

    def _active_enemies(self):
        return [e for e in self.enemies if e["active"] == 1.0]

    def _count_in_range(self):
        count = 0
        for enemy in self._active_enemies():
            if self._best_turret_for_enemy(enemy)[0] is not None:
                count += 1
        return count

    def _highest_threat_in_range(self):
        best = 0.0
        for enemy in self._active_enemies():
            if self._best_turret_for_enemy(enemy)[0] is not None:
                best = max(best, self._threat_score(enemy))
        return best

    def step(self, action):
        action = int(action)
        reward = 0.0
        self.last_shot = None
        self.last_hit = False
        self.last_impact = None

        in_range_count = self._count_in_range()
        best_range_threat = self._highest_threat_in_range()

        # -------- ACTION / SHOOTING --------
        if action < self.N_ENEMIES:
            target = self.enemies[action]

            if target["active"] == 0.0:
                reward -= 6.0
            else:
                turret_idx, turret_dist = self._best_turret_for_enemy(target)
                if turret_idx is None:
                    reward -= 4.0
                else:
                    self.turrets[turret_idx]["cooldown"] = self.COOLDOWN_TIME
                    pk = self._kill_probability(turret_dist)
                    self.last_shot = {
                        "turret": turret_idx,
                        "enemy": action,
                        "pk": pk,
                    }

                    threat = self._threat_score(target)
                    time_bonus = max(0.0, 18.0 - self._time_to_zone(target))
                    zone_bonus = self.zones[target["target_zone"]]["value"]

                    # reward simply for taking a valid shot
                    reward += 4.0 + 0.08 * threat

                    if self.rng.random() < pk:
                        target["active"] = 0.0
                        self.last_hit = True
                        reward += 25.0 + 0.7 * threat + 1.5 * time_bonus + 2.0 * zone_bonus
                    else:
                        reward -= 2.0 + 0.03 * threat
        else:
            # punish waiting when valid threats are in range
            if in_range_count > 0:
                reward -= 8.0 + 0.08 * best_range_threat
            else:
                reward += 0.1

        # -------- COOLDOWN UPDATE --------
        for turret in self.turrets:
            if turret["cooldown"] > 0:
                turret["cooldown"] -= 1

        # -------- ENEMY MOVEMENT / DAMAGE --------
        active_threats = 0
        for enemy in self.enemies:
            if enemy["active"] == 0.0:
                continue

            active_threats += 1
            enemy["pos"] += enemy["vel"] * self.DT

            dist_to_zone = self._zone_distance(enemy)
            time_to_zone = self._time_to_zone(enemy)
            zone_value = self.zones[enemy["target_zone"]]["value"]

            # stronger shaping: punish letting urgent threats get close
            reward -= max(0.0, 35.0 - dist_to_zone) * 0.06
            reward -= max(0.0, 12.0 - time_to_zone) * 0.35 * zone_value

            zone = self.zones[enemy["target_zone"]]
            if np.linalg.norm(enemy["pos"] - zone["pos"]) < zone["rad"]:
                enemy["active"] = 0.0
                damage = zone["value"] + enemy["damage_value"]
                self.total_damage += damage
                self.last_impact = {"enemy": enemy["id"], "zone": zone["type"], "damage": damage}

                # heavy penalty, especially for red zone
                reward -= 30.0 + 4.0 * damage + 3.0 * zone["value"]

        self.current_step += 1

        terminated = self.total_damage >= self.MAX_DAMAGE or (active_threats == 0 and self.current_step > 20)
        truncated = self.current_step >= self.MAX_STEPS

        info = {
            "damage_pct": float(self.total_damage),
            "active_enemies": int(sum(e["active"] == 1.0 for e in self.enemies)),
            "in_range_enemies": int(self._count_in_range()),
            "last_hit": self.last_hit,
            "last_impact": self.last_impact,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = []
        best_threat = 0.0
        active_count = 0
        in_range_count = self._count_in_range()

        for enemy in self.enemies:
            if enemy["active"] == 1.0:
                active_count += 1
                threat = self._threat_score(enemy)
                best_threat = max(best_threat, threat)
                zone_dist = self._zone_distance(enemy) / 70.0
                ttz = self._time_to_zone(enemy) / 30.0
                in_range = 1.0 if self._best_turret_for_enemy(enemy)[0] is not None else 0.0
            else:
                threat = 0.0
                zone_dist = 0.0
                ttz = 0.0
                in_range = 0.0

            obs.extend([
                enemy["pos"][0] / 60.0,
                enemy["pos"][1] / 60.0,
                enemy["pos"][2] / 40.0,
                enemy["vel"][0] / 4.0,
                enemy["vel"][1] / 4.0,
                enemy["vel"][2] / 4.0,
                enemy["active"],
                enemy["target_zone"] / max(1, len(self.zones) - 1),
                zone_dist,
                ttz,
                threat / 80.0,
                in_range,
            ])

        for turret in self.turrets:
            obs.append(turret["cooldown"] / max(1, self.COOLDOWN_TIME))

        obs.extend([
            self.total_damage / self.MAX_DAMAGE,
            self.current_step / self.MAX_STEPS,
            active_count / self.N_ENEMIES,
            in_range_count / self.N_ENEMIES,
            best_threat / 80.0,
            1.0 if in_range_count > 0 else 0.0,
        ])

        return np.array(obs, dtype=np.float32)