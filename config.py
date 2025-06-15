import os

# Game Settings
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BLOCK_SIZE = 20
BAR_HEIGHT = 40
SPEED = 150

# Colors (RGB)
COLOR_GREEN1 = (34, 139, 34)
COLOR_GREEN2 = (50, 205, 50)
COLOR_BG1 = (30, 30, 60)
COLOR_BG2 = (50, 50, 80)
COLOR_LIGHT_BORDER = (100, 130, 160)

# Training Hyperparameters
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9  # gamma
EPSILON_DECAY_START = 80  # Initial epsilon value
MAX_EPISODES = 500

# Plotting
PLOT_SAVE_EVERY = 20  # Episodes between saving plots

# File Paths
MODEL_FOLDER = './model'
BEST_MODEL_FILE = 'best_model.pth'
MODEL_FILE = 'model.pth'
BEST_SCORE_FILE = 'best_score.txt'  

# Model
ALGORITHM = 'ddqn' 

def model_path(filename, mode):
    return os.path.join(MODEL_FOLDER, f"{mode}_{filename}")