import numpy as np


class ClassicThreatPolicy:
    def predict(self, env):
        best_score = -1e9
        action = env.N_ENEMIES

        for i, enemy in enumerate(env.enemies):
            if enemy["active"] == 0.0:
                continue

            turret_idx, turret_dist = env._best_turret_for_enemy(enemy)
            if turret_idx is None:
                continue

            threat = env._threat_score(enemy)
            zone_time = env._time_to_zone(enemy)
            score = threat + max(0.0, 18.0 - zone_time) * 1.8 - turret_dist * 0.05

            if score > best_score:
                best_score = score
                action = i
        return action


class RandomPolicy:
    def predict(self, env):
        return np.random.randint(0, env.N_ENEMIES + 1)