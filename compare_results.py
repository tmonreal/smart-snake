import numpy as np
import matplotlib.pyplot as plt
import os

"""
Compare DQN vs DDQN vs Dueling DQN performance by loading scores and plotting results.
"""

def load_scores(mode):
    try:
        mean = np.load(f"model/results_{mode}_simple/mean_scores.npy", allow_pickle=True)
        scores = np.load(f"model/results_{mode}_simple/scores.npy", allow_pickle=True)
        return mean, scores
    except FileNotFoundError:
        print(f"❌ No se encontraron datos para {mode}")
        return [], []

def plot_comparison(mean_dqn, mean_ddqn, mean_dueling, scores_dqn, scores_ddqn, scores_dueling):
    # Mean score plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_dqn, label='DQN (media)', linewidth=2)
    plt.plot(mean_ddqn, label='DDQN (media)', linewidth=2)
    plt.plot(mean_dueling, label='Dueling DQN (media)', linewidth=2)
    plt.xlabel('Episodios')
    plt.ylabel('Score promedio')
    plt.title('Comparación DQN vs DDQN vs Dueling DQN (score promedio)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/comparacion_mean_scores_env_simple.png", dpi=300)

    # Raw score plot
    plt.figure(figsize=(10, 6))
    plt.plot(scores_dqn, label='DQN (score)', alpha=0.5)
    plt.plot(scores_ddqn, label='DDQN (score)', alpha=0.5)
    plt.plot(scores_dueling, label='Dueling DQN (score)', alpha=0.5)
    plt.xlabel('Episodios')
    plt.ylabel('Score individual')
    plt.title('Evolución de puntuación por episodio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/comparacion_scores_raw_env_simple.png", dpi=300)

if __name__ == "__main__":
    mean_dqn, scores_dqn = load_scores("dqn")
    mean_ddqn, scores_ddqn = load_scores("ddqn")
    mean_dueling, scores_dueling = load_scores("dueling")
    plot_comparison(mean_dqn, mean_ddqn, mean_dueling, scores_dqn, scores_ddqn, scores_dueling)