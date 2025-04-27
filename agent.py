import torch
import random
import os
import numpy as np
from collections import deque
import pygame
import sys
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from utils import plot, plot_convergence, load_best_score, save_best_score

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def menu():
    pygame.init()
    screen_width, screen_height = 640, 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Smart Snake by Trinidad Monreal - Choose Game Mode')
    font = pygame.font.Font('resources/PressStart2P-Regular.ttf', 12)

    # Colors
    BG_COLOR = (30, 30, 60)
    TEXT_COLOR = (255, 255, 255)
    BUTTON_COLOR = (50, 50, 80)       # Lighter blue tile
    BUTTON_HOVER_COLOR = (100, 130, 160)  # Light blue border

    # Buttons
    button_width, button_height = 400, 80
    button_train = pygame.Rect(120, 120, button_width, button_height)
    button_load = pygame.Rect(120, 240, button_width, button_height)

    clock = pygame.time.Clock()

    while True:
        screen.fill(BG_COLOR)

        # Mouse position
        mouse_pos = pygame.mouse.get_pos()

        # Draw buttons
        for button, text in [(button_train, "Start training from scratch"), (button_load, "Continue training best model")]: 
            if button.collidepoint(mouse_pos):
                pygame.draw.rect(screen, BUTTON_HOVER_COLOR, button, border_radius=15)
            else:
                pygame.draw.rect(screen, BUTTON_COLOR, button, border_radius=15)

            text_surf = font.render(text, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=button.center)
            screen.blit(text_surf, text_rect)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_train.collidepoint(event.pos):
                    return 1
                if button_load.collidepoint(event.pos):
                    return 2

        pygame.display.flip()
        clock.tick(60)

def train(agent, best_score, choice):
    if choice == 2:  # Load best model
        if os.path.exists('model/scores.npy'):
            plot_scores = np.load('model/scores.npy').tolist()
            plot_mean_scores = np.load('model/mean_scores.npy').tolist()
            successes = np.load('model/successes.npy').tolist()
            print(f"✅ Datos anteriores cargados: {len(plot_scores)} episodios previos.")
        else:
            plot_scores = []
            plot_mean_scores = []
            successes = []
            print("✅ No se encontraron datos anteriores, comenzando de cero.")
    else:  # Training from scratch
        plot_scores = []
        plot_mean_scores = []
        successes = []
        print("✅ Empezando entrenamiento desde cero. Gráficos reseteados.")

    total_score = 0
    record = best_score  # Local record for this session
    game = SnakeGameAI()
    game.set_session_record(record)
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save('model.pth')  # Save regular model
                game.set_session_record(record)

                # Check if this new session record beats the GLOBAL best
                if record > best_score:
                    best_score = record  # Update global best
                    agent.model.save('best_model.pth') 
                    save_best_score(best_score)  
                    game.flash_new_record()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            np.save('model/scores.npy', np.array(plot_scores))
            np.save('model/mean_scores.npy', np.array(plot_mean_scores))
            np.save('model/successes.npy', np.array(successes))

            success = 1 if score > 0 else 0
            successes.append(success)

            # Save the plot every 20 episodes
            if agent.n_games % 20 == 0:
                plot(plot_scores, plot_mean_scores, successes, save_path=f'plots/test3/plot_{agent.n_games}_record_{record}.png')
                plot_convergence(successes, moving_avg_window=20, save_path=f'plots/test3/convergence_{agent.n_games}_record_{record}.png')

def main():
    choice = menu()

    agent = Agent()

    if choice == 2:
        agent.model.load('best_model.pth') 
        best_score = load_best_score() 
        print(f"✅ Mejor modelo cargado con récord de {best_score} puntos.")
        train(agent, best_score)
    else:
        print("Entrenamiento nuevo iniciado.")
        best_score = 0

    train(agent, best_score, choice)

if __name__ == '__main__':
    main()