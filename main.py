import sys
import os
import torch
import pygame
import numpy as np
import config as cfg
from game import SmartSnake
from agent import SnakeAgent
from utils import plot_training_progress, plot_success_trend, load_best_score, save_best_score
from config import MAX_EPISODES

def menu():
    """
    Display a simple Pygame menu for the user to choose training mode.

    Returns:
        int: 1 if start from scratch, 2 if continue best model
    """
    pygame.init()
    screen_width, screen_height = 640, 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Smart Snake by Trinidad Monreal - Choose Game Mode')
    font = pygame.font.Font('resources/PressStart2P-Regular.ttf', 12)

    # Colors
    BG_COLOR = (30, 30, 60)
    TEXT_COLOR = (255, 255, 255)
    BUTTON_COLOR = (50, 50, 80)      
    BUTTON_HOVER_COLOR = (100, 130, 160)  

    # Button properties
    button_width, button_height = 400, 80
    button_train = pygame.Rect(120, 120, button_width, button_height)
    button_load = pygame.Rect(120, 240, button_width, button_height)

    clock = pygame.time.Clock()

    while True:
        screen.fill(BG_COLOR)
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

def train(snake_agent, best_score, choice, mode):
    """
    Main training loop for the Snake agent.

    Args:
        snake_agent (SnakeAgent): the agent to train
        best_score (int): best historical score
        choice (int): 1 for fresh training, 2 for continuing
    """
    # Initialize training metrics
    if choice == 2:  # Continue training best model
        if os.path.exists(f'model/results_{mode}/scores.npy'):
            plot_scores = np.load(f'model/results_{mode}/scores.npy').tolist()
            plot_mean_scores = np.load(f'model/results_{mode}/mean_scores.npy').tolist()
            successes = np.load(f'model/results_{mode}/successes.npy').tolist()
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
    game = SmartSnake()
    game.set_session_record(record)
    q_values_list = []
    
    while True:
        episode_q_values = [] 
        # 1. Get current state
        state_old = snake_agent.get_state(game)
        # 2. Get action from agent
        chosen_action = snake_agent.get_action(state_old)

        # Estimar valor máximo Q del estado actual
        with torch.no_grad():
            #q_values = snake_agent.model(torch.tensor(state_old, dtype=torch.float))
            q_values = snake_agent.model(torch.tensor(state_old, dtype=torch.float).unsqueeze(0))
            max_q = torch.max(q_values).item()
            episode_q_values.append(max_q)

        # 3. Perform action, get new state and reward
        reward, done, score = game.play_step(chosen_action)
        state_new = snake_agent.get_state(game)
        # 4. Train short memory (current step)
        snake_agent.train_on_step(state_old, chosen_action, reward, state_new, done)
        # 5. Store experience in replay buffer
        snake_agent.store_experience(state_old, chosen_action, reward, state_new, done)

        if done:
            if snake_agent.episodes_played >= MAX_EPISODES:
                print(f"🔚 Entrenamiento de prueba terminado ({MAX_EPISODES} episodios).")
                break
            # 6. Episode finished
            game.reset()
            game.set_episode(snake_agent.episodes_played + 1)
            snake_agent.episodes_played += 1
            snake_agent.train_replay_buffer()

            # Check if session record beaten
            if score > record:
                record = score
                snake_agent.model.save(cfg.model_path(cfg.MODEL_FILE, mode)) 
                game.set_session_record(record)

                # Check if global best beaten
                if record > best_score:
                    best_score = record  
                    snake_agent.model.save(cfg.model_path(cfg.BEST_MODEL_FILE, mode)) 
                    save_best_score(best_score, file_name=cfg.model_path(cfg.BEST_SCORE_FILE, mode))  
                    game.flash_new_record()

            print(f'Episode {snake_agent.episodes_played} | Score: {score} | Record: {record}')
            
            # 7. Update plots and metrics
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / snake_agent.episodes_played
            plot_mean_scores.append(mean_score)
            q_values_list.append(np.mean(episode_q_values))
            # Save plots data
            np.save(f'model/results_{mode}/scores.npy', np.array(plot_scores))
            np.save(f'model/results_{mode}/mean_scores.npy', np.array(plot_mean_scores))
            np.save(f'model/results_{mode}/successes.npy', np.array(successes))
            np.save(f'model/results_{mode}/q_values.npy', np.array(q_values_list))

            success = 1 if score > 0 else 0
            successes.append(success)

            # Plot progress every N episodes (defined in config.py)
            if snake_agent.episodes_played % cfg.PLOT_SAVE_EVERY == 0:
                os.makedirs(f'plots/{mode}', exist_ok=True)
                plot_training_progress(plot_scores, plot_mean_scores, successes, save_path=f'plots/{mode}/plot_{snake_agent.episodes_played}_record_{record}.png')
                plot_success_trend(successes, moving_avg_window=20, save_path=f'plots/{mode}/convergence_{snake_agent.episodes_played}_record_{record}.png')

def main(mode='ddqn'):
    """Main entry point of the program."""

    choice = menu()

    if mode == 'dqn':
        from model import DQLTrainer
        snake_agent = SnakeAgent(trainer_class=DQLTrainer)
    elif mode == 'ddqn':
        from model import DDQNTrainer
        snake_agent = SnakeAgent(trainer_class=DDQNTrainer)
    elif mode == 'dueling':
        from model import DuelingDQNTrainer
        snake_agent = SnakeAgent(trainer_class=DuelingDQNTrainer)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if choice == 2: # Load best model
        snake_agent.model.load(cfg.model_path(cfg.BEST_MODEL_FILE, mode))
        best_score = load_best_score(file_name=cfg.model_path(cfg.BEST_SCORE_FILE, mode))
        print(f"✅ Mejor modelo cargado con récord de {best_score} puntos.")
    else: # Start from scratch
        print("Entrenamiento nuevo iniciado.")
        best_score = 0

    train(snake_agent, best_score, choice, mode)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'ddqn'  # default is ddqn
    main(mode)