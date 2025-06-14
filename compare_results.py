import numpy as np
import matplotlib.pyplot as plt
import os

"""
Compare DQN vs DDQN performance by loading scores and plotting results.
# Entrenamiento DQN
python main.py dqn

# Entrenamiento DDQN
python main.py ddqn

# Comparación
python compare_results.py
"""
def load_scores(mode):
    try:
        mean = np.load(f"model/results_{mode}/mean_scores.npy", allow_pickle=True)
        scores = np.load(f"model/results_{mode}/scores.npy", allow_pickle=True)
        return mean, scores
    except FileNotFoundError:
        print(f"❌ No se encontraron datos para {mode}")
        return [], []

def plot_comparison(mean_dqn, mean_ddqn, scores_dqn, scores_ddqn):
    plt.figure(figsize=(10, 6))
    plt.plot(mean_dqn, label='DQN (media)', linewidth=2)
    plt.plot(mean_ddqn, label='DDQN (media)', linewidth=2)
    plt.xlabel('Episodios')
    plt.ylabel('Score promedio')
    plt.title('Comparación DQN vs DDQN')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/comparacion_mean_scores.png", dpi=300)
    plt.show()

    # También: gráfico de score puntual por episodio
    plt.figure(figsize=(10, 6))
    plt.plot(scores_dqn, label='DQN (score)', alpha=0.5)
    plt.plot(scores_ddqn, label='DDQN (score)', alpha=0.5)
    plt.xlabel('Episodios')
    plt.ylabel('Score individual')
    plt.title('Evolución de puntuación por episodio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/comparacion_scores_raw.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    mean_dqn, scores_dqn = load_scores("dqn")
    mean_ddqn, scores_ddqn = load_scores("ddqn")
    plot_comparison(mean_dqn, mean_ddqn, scores_dqn, scores_ddqn)