import torch
import copy
import random
import numpy as np
import config as cfg
from collections import deque
from game import Direction, Point
from model import DQLNetwork, DDQNTrainer, DQLTrainer, DuelingDQN

# Load hyperparameters from config
REPLAY_BUFFER_SIZE = cfg.REPLAY_BUFFER_SIZE
BATCH_SIZE = cfg.BATCH_SIZE
LR = cfg.LEARNING_RATE

class SnakeAgent:
    """
    Snake-playing agent trained with Double Deep Q-Learning (DDQN).

    - Maintains two networks: online (for action selection) and target (for evaluation).
    - Uses experience replay and epsilon-greedy strategy.
    - Periodically syncs target network to improve training stability.
    """
    def __init__(self, trainer_class):
        """Initialize the agent with model, trainer, and replay buffer."""
        self.episodes_played = 0
        self.epsilon = 0  # Exploration rate
        self.gamma = cfg.DISCOUNT_FACTOR
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)  # Experience replay memory

        input_size = 15  # State size
        hidden_size = 256
        output_size = 3  # Number of actions

        if trainer_class.__name__ == "DuelingDQNTrainer":
            self.is_ddqn = True
            self.model = DuelingDQN(input_size, hidden_size, output_size)
            self.target_model = DuelingDQN(input_size, hidden_size, output_size)
            self.trainer = trainer_class(self.model, self.target_model, lr=LR, gamma=self.gamma)
            print("Training with DuelingDQNTrainer.")

        elif trainer_class.__name__ == "DDQNTrainer":
            self.is_ddqn = True
            self.model = DQLNetwork(input_size, hidden_size, output_size)
            self.target_model = copy.deepcopy(self.model)
            self.trainer = trainer_class(self.model, self.target_model, lr=LR, gamma=self.gamma)
            print("Training with DDQNTrainer.")

        else:  
            self.is_ddqn = False
            self.model = DQLNetwork(input_size, hidden_size, output_size)
            self.trainer = trainer_class(self.model, lr=LR, gamma=self.gamma)
            print(f"Training with {trainer_class.__name__}.")

        self.update_target_network_every = 20


    def get_state(self, game):
        """
        Extract the current state representation from the game environment.
        
        Args:
            game (SmartSnake): current game environment
        
        Returns:
            np.array: binary array representing dangers, direction, and apple location
        """
        head = game.snake[0]

        # Compute surrounding points relative to head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Current moving direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Construct state array
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
            
            # Current move direction (one-hot encoding)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Apple relative position
            game.apple.x < game.head.x,  # Apple is left
            game.apple.x > game.head.x,  # Apple is right
            game.apple.y < game.head.y,  # Apple is up
            game.apple.y > game.head.y,  # Apple is down

            # Golden apple
            game.golden_apple.x < game.head.x if game.golden_apple else False,
            game.golden_apple.x > game.head.x if game.golden_apple else False,
            game.golden_apple.y < game.head.y if game.golden_apple else False,
            game.golden_apple.y > game.head.y if game.golden_apple else False,
            ]
            
        return np.array(state, dtype=int)

    def store_experience(self, state, action, reward, next_state, done):
        """Store a new experience tuple in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done)) 

    def train_replay_buffer(self):
        """Train the model by sampling a batch from the replay buffer."""
        if len(self.replay_buffer) > BATCH_SIZE:
            mini_sample = random.sample(self.replay_buffer, BATCH_SIZE) # Random batch
        else:
            mini_sample = self.replay_buffer # Not enough samples yet

        # Unpack mini-batch
        states, actions, rewards, next_states, dones = zip(*[
            (s, np.argmax(a), r, ns, d) for (s, a, r, ns, d) in mini_sample
        ])


        # Train the model using the batch
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # Update target network periodically
        if self.is_ddqn and self.episodes_played % self.update_target_network_every == 0:
            self.update_target_network()


    def train_on_step(self, state, action, reward, next_state, done):
        """Train the model immediately on a single step."""
        idx_action = np.argmax(action)  # transform from one-hot to index
        self.trainer.train_step(state, idx_action, reward, next_state, done)
        #self.trainer.train_step(state, action, reward, next_state, done)

    def update_epsilon(self):
        """Update the epsilon value based on episodes played."""
        self.epsilon = max(0, cfg.EPSILON_DECAY_START - self.episodes_played)


    def get_action(self, state):
        """
        Choose an action based on epsilon-greedy strategy.
        
        Args:
            state (np.array): current environment state
        
        Returns:
            list: chosen action as one-hot encoded [straight, right, left]
        """
        self.update_epsilon() # Linearly decay exploration
        chosen_action = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            # Exploration: choose random action
            action = random.randint(0, 2)
            chosen_action[action] = 1
        else:
            # Exploitation: choose best action predicted by model
            #state0 = torch.tensor(state, dtype=torch.float)
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(state0)
            action = torch.argmax(prediction).item()
            chosen_action[action] = 1

        return chosen_action
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"[Episode {self.episodes_played}] Target network updated.")