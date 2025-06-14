import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_progress(scores, mean_scores, successes, save_path=None):
    """
    Plot the training progress showing score and success rate.

    Args:
        scores (list): list of individual episode scores.
        mean_scores (list): list of running mean scores.
        successes (list): list indicating success (1) or failure (0) per episode.
        save_path (str, optional): path to save the figure. If None, does not save.
    """
    plt.clf()
    plt.style.use('seaborn-v0_8-darkgrid') 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8)) 

    # -- First graph: Score Progress --
    ax1.set_title('Entrenamiento - Recompensa', fontsize=16)
    ax1.set_xlabel('Episodios', fontsize=12)
    ax1.set_ylabel('Recompensa', fontsize=12)
    ax1.plot(scores, label='Recompensa', color='tab:blue', linewidth=2)
    ax1.plot(mean_scores, label='Recompensa Media', color='tab:orange', linewidth=2)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    # -- Second graph: Success Rate Progress --
    success_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
    ax2.set_title('Entrenamiento - Tasa de Éxito', fontsize=16)
    ax2.set_xlabel('Episodios', fontsize=12)
    ax2.set_ylabel('Tasa de Éxito', fontsize=12)
    ax2.plot(success_rate, label='Tasa de Éxito', color='tab:green', linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_trend(successes, moving_avg_window=20, save_path=None):
    """
    Plot the success trend using a moving average.

    Args:
        successes (list): list of success indicators (1 for success, 0 for fail).
        moving_avg_window (int): size of the window for moving average.
        save_path (str, optional): path to save the figure. If None, does not save.
    """
    plt.style.use('seaborn-v0_8-darkgrid')  
    
    if len(successes) >= moving_avg_window:
        success_rate_all = successes
        moving_avg_success = np.convolve(success_rate_all, np.ones(moving_avg_window)/moving_avg_window, mode='valid')

        plt.figure(figsize=(10, 6))  
        plt.plot(np.arange(moving_avg_window - 1, len(successes)), moving_avg_success * 100,
                 color='tab:green', linewidth=2, label=f'Media móvil ({moving_avg_window} episodios)')
        plt.title(f'Tasa de Éxito (Media Móvil {moving_avg_window} Episodios)', fontsize=16)
        plt.xlabel('Episodio', fontsize=12)
        plt.ylabel('Tasa de Éxito (%)', fontsize=12)
        plt.grid(True)
        plt.ylim(-5, 105)
        plt.legend(fontsize=10)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    else:
        # If not enough episodes yet, plot raw success curve
        print("No hay suficientes episodios para calcular la media móvil.")
        plt.figure(figsize=(10, 6))
        plt.plot(successes, color='tab:blue', linewidth=2, label='Éxito por Episodio')
        plt.title('Éxito por Episodio', fontsize=16)
        plt.xlabel('Episodio', fontsize=12)
        plt.ylabel('Éxito (1=Sí, 0=No)', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=10)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def save_best_score(score, file_name='model/best_score.txt'):
    """
    Save the best score to a text file.

    Args:
        score (int): best score to save.
        file_name (str): file path to save the score.
    """
    with open(file_name, 'w') as f:
        f.write(str(score))

def load_best_score(file_name='model/best_score.txt'):
    """
    Load the best score from a text file.

    Args:
        file_name (str): file path to load the score from.

    Returns:
        int: best score loaded, 0 if file not found.
    """
    try:
        with open(file_name, 'r') as f:
            return int(f.read())
    except FileNotFoundError:
        return 0  # If no best_score.txt yet, assume record = 0

