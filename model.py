import numpy as np
import config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQLNetwork(nn.Module):
    """Deep Q-Learning Network: simple 2-layer fully connected model."""

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network layers.
        
        Args:
            input_size (int): number of input neurons (state size)
            hidden_size (int): number of hidden layer neurons
            output_size (int): number of output neurons (action space size)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): input state
        
        Returns:
            torch.Tensor: Q-values for each possible action
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name=cfg.MODEL_FILE):
        """
        Save model parameters to file.
        
        Args:
            file_name (str): filename to save the model to
        """
        model_folder_path = cfg.MODEL_FOLDER
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print("💾 Modelo guardado correctamente en", file_path)

    def load(self, file_name=cfg.MODEL_FILE):
        """
        Load model parameters from file.
        
        Args:
            file_name (str): filename to load the model from
        """
        model_folder_path = cfg.MODEL_FOLDER
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            print("⏳ Modelo cargado correctamente desde", file_path)
        else:
            print("⚠️ No se encontró un modelo guardado.")


class DQLTrainer:
    """Trainer class for Deep Q-Learning agent using MSE loss and Adam optimizer."""

    def __init__(self, model, lr, gamma):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Q-network to be trained
            lr (float): learning rate
            gamma (float): discount factor
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Train the model on a batch or single experience.
        
        Args:
            state (torch.Tensor or np.array): current state(s)
            action (torch.Tensor or np.array): action(s) taken
            reward (torch.Tensor or np.array): reward(s) received
            next_state (torch.Tensor or np.array): next state(s)
            done (bool or list of bools): whether the episode ended
        """
        # Convert to torch tensors if needed
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        # Handle single sample dimensions
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q-values for current state
        pred = self.model(state)
        # Create target tensor
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()