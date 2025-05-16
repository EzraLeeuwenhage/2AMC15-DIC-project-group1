"""
THIS FIGURE IS FOR ILLUSTRATIVE PURPOSES ONLY. IT WAS LARGELY CREATED USING CHATGPT.
THIS WAS USEFUL FOR GENERATING VISUALLY PLEASING SYNTHETIC DATA AND PLOTTING IT.
"""

import numpy as np
import matplotlib.pyplot as plt

# Synthetic example data for nicely and clearly visualizing all metrics
n_runs = 20
n_episodes = 100
min_reward = 0.0
max_reward = 1.0
episodes = np.arange(1, n_episodes + 1)

# Generate synthetic reward curves with varying learning speeds and noise
runs = []
for _ in range(n_runs):
    tau = np.random.uniform(10, 30)  # different time constants
    base_curve = max_reward * (1 - np.exp(-episodes / tau))
    noise = np.random.normal(scale=0.1, size=n_episodes)  # higher noise for diversity
    run = np.clip(base_curve + noise, min_reward, max_reward)
    runs.append(run)
runs = np.array(runs)

# Compute expected (mean) and lower confidence bound (mean - std)
mean_reward = runs.mean(axis=0)
std_reward = runs.std(axis=0)
lcb_reward = mean_reward - std_reward

# Normalized AUC calculation
def calc_normalized_auc(reward_list, min_r, max_r):
    area_above_min = np.sum(reward_list - min_r)
    total_area = (max_r - min_r) * len(reward_list)
    return area_above_min / total_area

auc_value = calc_normalized_auc(mean_reward, min_reward, max_reward)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(right=0.8)

for run in runs:
    ax.plot(episodes, run, color='gray', alpha=0.3)

# Mean and LCB
ax.plot(episodes, mean_reward, color='blue', linewidth=2, label='Expected Reward')
ax.plot(episodes, lcb_reward, color='red', linewidth=2, linestyle='--', label='Lower Confidence Bound')

# Theoretical (min-max scaled) bounds
ax.hlines(min_reward, episodes[0], episodes[-1], linestyle=':', color='black',
          label='Normalized theoretical min')
ax.hlines(max_reward, episodes[0], episodes[-1], linestyle=':', color='black',
          label='Normalized theoretical max')

# Highlight maxima with circles only
mean_max_idx = np.argmax(mean_reward)
lcb_max_idx = np.argmax(lcb_reward)

ax.scatter([episodes[mean_max_idx]], [mean_reward[mean_max_idx]],
           s=200, facecolors='none', edgecolors='blue', linewidths=2)
ax.scatter([episodes[lcb_max_idx]], [lcb_reward[lcb_max_idx]],
           s=200, facecolors='none', edgecolors='red', linewidths=2)

# Shade area under expected curve above min_reward
ax.fill_between(episodes, min_reward, mean_reward, color='blue', alpha=0.1)

# Annotate AUC
ax.text(0.05 * n_episodes, min_reward + 0.9 * (max_reward - min_reward),
        f"Normalized AUC = {auc_value:.2f}", fontsize=12)

ax.set_xlabel('Episodes')
ax.set_ylabel('Cumulative Reward')
ax.set_title('Reward Curves with Expected, LCB, and Theoretical Bounds')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
