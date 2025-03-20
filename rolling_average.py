import pickle
import matplotlib.pyplot as plt
import numpy as np

# Function to compute rolling average
def rolling_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# **Load DQN Results**
try:
    with open("dqn_rewards.pkl", "rb") as f:
        dqn_rewards, dqn_lengths = pickle.load(f)
    print("✅ Loaded DQN results successfully.")
except FileNotFoundError:
    print("❌ DQN results file not found! Ensure `dqn.py` has been run.")
    dqn_rewards, dqn_lengths = None, None

# **Load Actor-Critic Results**
try:
    with open("actor_critic_rewards.pkl", "rb") as f:
        ac_data = pickle.load(f)
    
    if isinstance(ac_data, tuple) and len(ac_data) == 2:
        ac_rewards, ac_lengths = ac_data
    else:
        ac_rewards = ac_data  # Assume only rewards exist
        ac_lengths = None  # Set lengths to None if missing
    
    print("✅ Loaded Actor-Critic results successfully.")
except FileNotFoundError:
    print("❌ Actor-Critic results file not found! Ensure `actor_critic.py` has been run.")
    ac_rewards, ac_lengths = None, None

# **Check if Both Results Exist**
if dqn_rewards is None or ac_rewards is None:
    print("❌ Comparison cannot be performed due to missing data.")
    exit()

# **Compute Rolling Averages**
window_size = 5
dqn_rolling_avg = rolling_average(dqn_rewards, window_size)
ac_rolling_avg = rolling_average(ac_rewards, window_size)

# **Plot Training Progress with Rolling Average**
plt.figure(figsize=(12, 6))

plt.plot(dqn_rolling_avg, label="DQN (Rolling Avg)", linestyle="-", marker="o", markersize=3)
plt.plot(ac_rolling_avg, label="Actor-Critic (Rolling Avg)", linestyle="-", marker="s", markersize=3)

plt.xlabel("Episode")
plt.ylabel("Rolling Avg Reward (Window=5)")
plt.title("DQN vs Actor-Critic Training Progress in Pac-Man (Rolling Avg)")
plt.legend()
plt.grid(True)

plt.savefig("compare_dqn_vs_ac_rolling_avg.png")
plt.show()
