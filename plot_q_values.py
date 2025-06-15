import numpy as np
import matplotlib.pyplot as plt
import os

def moving_avg(values, window):
    return np.convolve(values, np.ones(window)/window, mode='valid')

# Cargar Q-values estimados
q_dqn = np.load("model/results_dqn/q_values.npy", allow_pickle=True)
q_ddqn = np.load("model/results_ddqn/q_values.npy", allow_pickle=True)

# Suavizado
q_dqn_smooth = moving_avg(q_dqn, window=100)
q_ddqn_smooth = moving_avg(q_ddqn, window=100)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(q_dqn_smooth, label="DQN Estimado (max Q)", color='orange', linewidth=2)
plt.plot(q_ddqn_smooth, label="DDQN Estimado (max Q)", color='blue', linewidth=2)

plt.title("Comparación Estimación de Q-Values")
plt.xlabel("Episodio")
plt.ylabel("Valor Q Estimado Promedio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/comparacion_q_values.png", dpi=300)
plt.show()