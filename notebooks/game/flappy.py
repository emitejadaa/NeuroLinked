import pygame
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import load_model
import os

# --------------------- Modelo de IA ---------------------
class model:
    def __init__(self, mutation_rate=0.005):
        self.mutation_rate = mutation_rate
        self.model = Sequential([
            layers.Dense(32, activation="relu", input_shape=(8,)),
            layers.Dense(1, activation="sigmoid")
        ])

    def mutate(self):
        new_weights = []
        for w in self.model.get_weights():
            noise = np.random.normal(0, self.mutation_rate, w.shape)
            new_w = w + noise
            new_weights.append(new_w)
        self.model.set_weights(new_weights)


# --------------------- Obstáculos ---------------------
class Obstacle:
    def __init__(self, x, width, gap, screen_height):
        self.x = x
        self.width = width
        self.gap = gap
        self.screen_height = screen_height
        self.top_height = random.randint(50, screen_height - gap - 50)
        self.bottom_height = screen_height - self.top_height - gap
        self.color = (0, 255, 0)
        self.passed = False

    def update(self, speed):
        self.x -= speed

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, 0, self.width, self.top_height))
        pygame.draw.rect(screen, self.color, (self.x, self.screen_height - self.bottom_height, self.width, self.bottom_height))

    def off_screen(self):
        return self.x + self.width < 0

    def get_rects(self):
        top_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        bottom_rect = pygame.Rect(self.x, self.screen_height - self.bottom_height, self.width, self.bottom_height)
        return top_rect, bottom_rect


# --------------------- Juego ---------------------
class Connect:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 650))
        pygame.display.set_caption("Flappy AI Demo")
        self.clock = pygame.time.Clock()
        self.running = True
        self.gravity = 0.5
        self.players = self.createPlayers(10)
        self.generation = 1

        self.score = 0
        self.font = pygame.font.SysFont("Arial", 32)

        self.obstacles = []
        self.spawn_timer = 0
        self.obstacle_gap = 150
        self.obstacle_width = 60
        self.obstacle_speed = 3

    # --------------------- Ciclo principal ---------------------
    def run(self):
        while self.running:
            self.screen.fill((255, 255, 255))
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    if event.key == pygame.K_SPACE and self.players:
                        self.jump(0)

            self.update(self.players)
            self.update_obstacles()
            dead_players = self.check_collisions()

            self.check_score()
            self.drawPlayers(self.players)
            self.draw_obstacles()
            self.draw_score()

            pygame.display.flip()

            for p in self.players:
                p["fitness"] += 0.01

            if len(self.players) == 0:
                print("Todos los jugadores murieron!")
                self.evolve(dead_players)

        pygame.quit()
        sys.exit()

    # --------------------- Estado de jugador ---------------------
    def get_player_state(self, player, num_future_obstacles=1):
        screen_height = 650
        player_y = player["player_y"] / screen_height
        player_vel = player["player_vel"] / 20.0
        player_size = player["player_size"] / 100.0
        state = [player_y, player_vel, player_size]

        sorted_obs = sorted(
            [obs for obs in self.obstacles if obs.x + obs.width >= player["player_x"]],
            key=lambda o: o.x
        )

        for i in range(num_future_obstacles):
            if i < len(sorted_obs):
                obs = sorted_obs[i]
                dist_x = (obs.x + obs.width - player["player_x"]) / 600.0
                top_height = obs.top_height / screen_height
                bottom_height = obs.bottom_height / screen_height
                gap_size = obs.gap / screen_height
                gap_center = (obs.top_height + obs.gap / 2) / screen_height
                delta_y = (player["player_y"] / screen_height - gap_center)
            else:
                dist_x = 1.0
                top_height = 0.0
                bottom_height = 0.0
                gap_size = 0.0
                delta_y = 0.0
            state += [dist_x, top_height, bottom_height, gap_size, delta_y]

        return state

    # --------------------- Jugadores ---------------------
    def createPlayers(self, n_players):
        players = []
        best_model_exists = os.path.exists("best_player.h5")

        for i in range(n_players):
            brain = model()
            if best_model_exists:
                brain.model = load_model("best_player.h5")
            players.append({
                "player_size": 40,
                "player_x": 50,
                "player_y": 300,
                "player_speed": 5,
                "player_color": (255, 0, 0),
                "player_n": i + 1,
                "player_vel": 0,
                "player_jump": -8,
                "brain": brain,
                "fitness": 0,
                "jump_cooldown": 0
            })
        return players

    def drawPlayers(self, players):
        for p in players:
            pygame.draw.rect(self.screen, p["player_color"], (p["player_x"], p["player_y"], p["player_size"], p["player_size"]))

    # --------------------- Actualizar jugadores ---------------------
    def update(self, players):
        for i, p in enumerate(players):
            p["player_vel"] += self.gravity
            p["player_y"] += p["player_vel"]

            if p["jump_cooldown"] > 0:
                p["jump_cooldown"] -= 1

            state = np.array(self.get_player_state(p)).reshape(1, -1)
            action = p["brain"].model(state).numpy()[0][0]

            if action > 0.5 and p["jump_cooldown"] == 0:
                self.jump(i)
                p["jump_cooldown"] = 5

    def jump(self, player):
        self.players[player]["player_vel"] = self.players[player]["player_jump"]

    # --------------------- Obstáculos ---------------------
    def spawn_obstacle(self):
        self.obstacles.append(Obstacle(600, self.obstacle_width, self.obstacle_gap, 650))

    def update_obstacles(self):
        self.spawn_timer += 1
        if self.spawn_timer > 90:
            self.spawn_obstacle()
            self.spawn_timer = 0

        for obs in self.obstacles:
            obs.update(self.obstacle_speed)

        self.obstacles = [obs for obs in self.obstacles if not obs.off_screen()]

    def draw_obstacles(self):
        for obs in self.obstacles:
            obs.draw(self.screen)

    # --------------------- Puntuación ---------------------
    def check_score(self):
        for obs in self.obstacles:
            if not obs.passed and self.players:
                if self.players[0]["player_x"] > obs.x + obs.width:
                    obs.passed = True
                    for p in self.players:
                        p["fitness"] += 1
                    self.score += 1

    # --------------------- Puntuación en pantalla ---------------------
    def draw_score(self):
        score_surface = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_surface, (10, 10))

        gen_surface = self.font.render(f"Generation: {self.generation}", True, (0, 0, 0))
        self.screen.blit(gen_surface, (10, 50))

        alive_surface = self.font.render(f"Players Alive: {len(self.players)}", True, (0, 0, 0))
        self.screen.blit(alive_surface, (10, 90))

    # --------------------- Colisiones ---------------------
    def check_collisions(self):
        remaining_players = []
        dead_players = []
        for p in self.players:
            player_rect = pygame.Rect(p["player_x"], p["player_y"], p["player_size"], p["player_size"])
            collision = False
            for obs in self.obstacles:
                top_rect, bottom_rect = obs.get_rects()
                if player_rect.colliderect(top_rect) or player_rect.colliderect(bottom_rect):
                    collision = True
                    break
            if p["player_y"] <= 0 or p["player_y"] + p["player_size"] >= 650:
                collision = True

            if not collision:
                remaining_players.append(p)
            else:
                dead_players.append(p)

        self.players = remaining_players
        return dead_players

    # --------------------- Evolución jerárquica ---------------------
    def evolve(self, dead_players):
        all_players = dead_players + self.players
        all_players.sort(key=lambda p: p["fitness"], reverse=True)

        # Guardar el mejor modelo
        best_player = all_players[0]
        best_player["brain"].model.save("best_player.h5")

        # Definir hijos por ranking
        rank_children = [5,5,4,4,3,2,1]  # para los 7 mejores
        new_players = []

        # Crear nuevos jugadores según ranking
        for rank, n_children in enumerate(rank_children):
            if rank >= len(all_players):
                break
            parent_brain = all_players[rank]["brain"]
            for _ in range(n_children):
                child = model()
                child.model.set_weights(parent_brain.model.get_weights())
                child.mutate()
                new_players.append({
                    "player_size": 40,
                    "player_x": 50,
                    "player_y": 300,
                    "player_speed": 5,
                    "player_color": (255, 0, 0),
                    "player_n": len(new_players)+1,
                    "player_vel": 0,
                    "player_jump": -8,
                    "brain": child,
                    "fitness": 0,
                    "jump_cooldown": 0
                })

        # Mantener al mejor jugador sin cambios
        new_players.append({
            "player_size": 40,
            "player_x": 50,
            "player_y": 300,
            "player_speed": 5,
            "player_color": (0, 0, 255),  # destacar al mejor
            "player_n": len(new_players)+1,
            "player_vel": 0,
            "player_jump": -8,
            "brain": best_player["brain"],
            "fitness": 0,
            "jump_cooldown": 0
        })

        print(f"Generación {self.generation} iniciando...")
        self.generation += 1
        self.players = new_players
        self.obstacles = []
        self.score = 0


