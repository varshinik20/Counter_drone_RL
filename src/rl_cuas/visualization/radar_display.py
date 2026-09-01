print("--------------------------------------------------")
print("VISUALIZER SYSTEM STARTUP...")
print("--------------------------------------------------")

import copy
import math
from pathlib import Path

import pygame
import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import PPO

from ..env import SwarmDefenseEnv
from ..config import ScenarioConfig
from ..policies.baselines import ClassicThreatPolicy

WIDTH, HEIGHT = 1400, 600
BG_COLOR = (0, 0, 0)
GRID_COLOR = (40, 50, 60)
ZONE_RED = (150, 0, 0)
ZONE_ORANGE = (150, 100, 0)
ENEMY_COLOR = (0, 255, 255)
TURRET_COLOR = (255, 255, 0)
BEAM_COLOR = (255, 255, 120)
HIT_COLOR = (0, 255, 120)
WARN_COLOR = (255, 80, 80)
FLASH_COLOR = (255, 255, 255)

SCALE = 5.0
ISO_ANGLE = math.radians(30)


class DualVisualizer:
    def __init__(self, model_path="results/models/commander_policy"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Counter UAS - Classic vs AI Benchmark")
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("eurostile", 30, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 20)
        self.font_xs = pygame.font.SysFont("consolas", 16)

        cfg = ScenarioConfig()
        self.env_ai = SwarmDefenseEnv(cfg, seed=42)
        self.env_classic = SwarmDefenseEnv(cfg, seed=42)
        self.classic_policy = ClassicThreatPolicy()

        try:
            self.model = PPO.load(model_path, env=self.env_ai)
            print("AI model loaded successfully.")
        except Exception as e:
            print(f"Model load error: {e}")
            self.model = None

        self.lasers_ai = []
        self.lasers_classic = []
        self.ai_status = "READY"
        self.classic_status = "READY"

        # -------- VIDEO RECORDER --------
        self.record = True
        self.video_writer = None

        if self.record:
            Path("results").mkdir(exist_ok=True)
            self.video_writer = imageio.get_writer(
                "results/demo.mp4",
                fps=30,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p"
            )

    def project(self, pos, offset_x):
        x, y, z = pos[0], pos[1], pos[2]
        rot_x = (x - y) * math.cos(ISO_ANGLE)
        rot_y = (x + y) * math.sin(ISO_ANGLE) - z
        sx = offset_x + int(rot_x * SCALE)
        sy = (HEIGHT // 2 + 100) + int(rot_y * SCALE)
        return sx, sy

    def draw_scene(self, env, offset_x, title, lasers, status_text):
        pygame.draw.rect(self.screen, (10, 10, 10), (offset_x - 310, 35, 620, 470), border_radius=12)

        for i in range(-50, 51, 10):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                self.project([i, -50, 0], offset_x),
                self.project([i, 80, 0], offset_x),
                1,
            )
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                self.project([-50, i, 0], offset_x),
                self.project([50, i, 0], offset_x),
                1,
            )

        corners = [
            [-40, 50, 10], [40, 50, 10], [40, 60, 10], [-40, 60, 10],
            [-40, 50, 30], [40, 50, 30], [40, 60, 30], [-40, 60, 30]
        ]
        cp = [self.project(p, offset_x) for p in corners]
        pygame.draw.lines(self.screen, (120, 120, 120), True, cp[:4], 1)
        pygame.draw.lines(self.screen, (120, 120, 120), True, cp[4:], 1)
        for i in range(4):
            pygame.draw.line(self.screen, (120, 120, 120), cp[i], cp[i + 4], 1)

        for zone in env.zones:
            col = ZONE_RED if zone["type"] == "red" else ZONE_ORANGE
            center = self.project(zone["pos"], offset_x)
            rect = pygame.Rect(
                center[0] - zone["rad"] * SCALE,
                center[1] - zone["rad"] * SCALE * 0.5,
                zone["rad"] * SCALE * 2,
                zone["rad"] * SCALE
            )
            pygame.draw.ellipse(self.screen, col, rect)
            pygame.draw.ellipse(self.screen, (255, 200, 0), rect, 1)

        for turret in env.turrets:
            pt = self.project(turret["pos"], offset_x)
            pygame.draw.polygon(
                self.screen,
                TURRET_COLOR,
                [(pt[0], pt[1] - 5), (pt[0] - 4, pt[1] + 3), (pt[0] + 4, pt[1] + 3)]
            )

        for enemy in env.enemies:
            if enemy["active"] > 0:
                ep = self.project(enemy["pos"], offset_x)
                s = 3
                pygame.draw.line(self.screen, ENEMY_COLOR, (ep[0] - s, ep[1] - s), (ep[0] + s, ep[1] + s), 2)
                pygame.draw.line(self.screen, ENEMY_COLOR, (ep[0] - s, ep[1] + s), (ep[0] + s, ep[1] - s), 2)
                gp = self.project([enemy["pos"][0], enemy["pos"][1], 0], offset_x)
                pygame.draw.line(self.screen, (30, 30, 30), gp, ep, 1)

        for l in lasers[:]:
            l["f"] -= 1
            if l["f"] <= 0:
                lasers.remove(l)
                continue

            pygame.draw.line(self.screen, BEAM_COLOR, l["s"], l["e"], 4)
            pygame.draw.circle(self.screen, FLASH_COLOR, l["e"], 6)
            pygame.draw.circle(self.screen, BEAM_COLOR, l["e"], 10, 2)

        title_surf = self.font_lg.render(title, True, (255, 255, 255))
        self.screen.blit(title_surf, (offset_x - title_surf.get_width() // 2, HEIGHT - 105))

        damage_color = HIT_COLOR if env.total_damage < 30 else (255, 190, 0) if env.total_damage < 50 else WARN_COLOR
        dmg_surf = self.font_sm.render(f"Damage: {env.total_damage:.2f} %", True, damage_color)
        active_count = sum(e["active"] for e in env.enemies)
        active_surf = self.font_xs.render(f"Active threats: {int(active_count)}", True, (200, 200, 200))
        range_surf = self.font_xs.render(f"In-range threats: {env._count_in_range()}", True, (200, 200, 200))
        status_surf = self.font_xs.render(f"Status: {status_text}", True, (255, 255, 100))

        self.screen.blit(dmg_surf, (offset_x - dmg_surf.get_width() // 2, HEIGHT - 72))
        self.screen.blit(active_surf, (offset_x - active_surf.get_width() // 2, HEIGHT - 48))
        self.screen.blit(range_surf, (offset_x - range_surf.get_width() // 2, HEIGHT - 28))
        self.screen.blit(status_surf, (offset_x - status_surf.get_width() // 2, HEIGHT - 8))

    def reset_pair(self):
        obs_ai, _ = self.env_ai.reset(seed=42)
        self.env_classic.reset(seed=42)
        self.env_classic.enemies = copy.deepcopy(self.env_ai.enemies)
        self.env_classic.total_damage = 0.0
        self.env_classic.current_step = 0
        self.lasers_ai.clear()
        self.lasers_classic.clear()
        self.ai_status = "READY"
        self.classic_status = "READY"
        return obs_ai

    def run(self):
        obs_ai = self.reset_pair()
        running = True

        while running:
            self.screen.fill(BG_COLOR)
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # AI
            if self.model is not None:
                action_ai, _ = self.model.predict(obs_ai, deterministic=True)
            else:
                action_ai = self.env_ai.N_ENEMIES

            obs_ai, _, term_ai, trunc_ai, info_ai = self.env_ai.step(action_ai)
            self.ai_status = "IDLE"

            if self.env_ai.last_shot:
                ti = self.env_ai.last_shot["turret"]
                ei = self.env_ai.last_shot["enemy"]
                start = self.project(self.env_ai.turrets[ti]["pos"], 3 * WIDTH // 4)
                end = self.project(self.env_ai.enemies[ei]["pos"], 3 * WIDTH // 4)
                self.lasers_ai.append({"s": start, "e": end, "f": 12})
                self.ai_status = "SHOT FIRED"

            if info_ai.get("last_hit"):
                self.ai_status = "TARGET HIT"
            elif info_ai.get("last_impact") is not None:
                self.ai_status = "ZONE IMPACT"

            # Classic
            action_classic = self.classic_policy.predict(self.env_classic)
            _, _, term_classic, trunc_classic, info_classic = self.env_classic.step(action_classic)
            self.classic_status = "IDLE"

            if self.env_classic.last_shot:
                ti = self.env_classic.last_shot["turret"]
                ei = self.env_classic.last_shot["enemy"]
                start = self.project(self.env_classic.turrets[ti]["pos"], WIDTH // 4)
                end = self.project(self.env_classic.enemies[ei]["pos"], WIDTH // 4)
                self.lasers_classic.append({"s": start, "e": end, "f": 12})
                self.classic_status = "SHOT FIRED"

            if info_classic.get("last_hit"):
                self.classic_status = "TARGET HIT"
            elif info_classic.get("last_impact") is not None:
                self.classic_status = "ZONE IMPACT"

            banner = self.font_lg.render("Counter UAS - Anti Swarm System", True, (255, 255, 255))
            subtitle = self.font_sm.render("Scenario A: 3 Sensitive Areas - 4 Kinetic Effectors", True, (220, 220, 220))
            self.screen.blit(banner, (WIDTH // 2 - banner.get_width() // 2, 10))
            self.screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 45))

            self.draw_scene(self.env_classic, WIDTH // 4, "Classic Control", self.lasers_classic, self.classic_status)
            self.draw_scene(self.env_ai, 3 * WIDTH // 4, "AI-Based Control", self.lasers_ai, self.ai_status)
            pygame.draw.line(self.screen, (50, 50, 50), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)

            pygame.display.flip()

            # -------- RECORD FRAME --------
            if self.record and self.video_writer is not None:
                frame = pygame.surfarray.array3d(self.screen)
                frame = np.transpose(frame, (1, 0, 2))
                self.video_writer.append_data(frame)

            if term_ai or trunc_ai or term_classic or trunc_classic:
                pygame.time.delay(1200)
                obs_ai = self.reset_pair()

        # -------- RELEASE VIDEO --------
        if self.record and self.video_writer is not None:
            self.video_writer.close()

        pygame.quit()


if __name__ == "__main__":
    DualVisualizer().run()