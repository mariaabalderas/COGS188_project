import pickle
import matplotlib.pyplot as plt
import numpy as np

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

    # Check if the data is a tuple with two elements
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

# **Compute Statistics**
dqn_avg_reward = np.mean(dqn_rewards)
ac_avg_reward = np.mean(ac_rewards)

dqn_best = np.max(dqn_rewards)
ac_best = np.max(ac_rewards)

print(f"\n📊 Performance Summary:")
print(f"DQN -> Avg Reward: {dqn_avg_reward:.2f}, Best Reward: {dqn_best}")
print(f"Actor-Critic -> Avg Reward: {ac_avg_reward:.2f}, Best Reward: {ac_best}")

# **Plot Training Progress**
plt.figure(figsize=(12, 6))

plt.plot(dqn_rewards, label="DQN", linestyle="-", marker="o", markersize=3)
plt.plot(ac_rewards, label="Actor-Critic", linestyle="-", marker="s", markersize=3)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN vs Actor-Critic Training Progress in Pac-Man")
plt.legend()
plt.grid(True)

plt.savefig("compare_dqn_vs_ac.png")
plt.show()