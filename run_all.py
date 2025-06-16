import subprocess

# Training runs
print("1. Starting DQN training...")
subprocess.run(["python", "main.py", "dqn"])

print("2. Starting DDQN training...")
subprocess.run(["python", "main.py", "ddqn"])

print("3. Starting Dueling DQN training...")
subprocess.run(["python", "main.py", "dueling"])

# Post-training analysis
print("Generating score comparisons...")
subprocess.run(["python", "compare_results.py"])

print("Generating Q-values comparison...")
subprocess.run(["python", "plot_q_values.py"])

print("All tasks completed!")