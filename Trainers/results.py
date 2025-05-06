import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your results
df = pd.read_csv("tune_results.csv")

# 2. Identify your reward columns
reward_cols = [c for c in df.columns if "reward" in c.lower()]
print("Reward columns:", reward_cols)

# 3. (Optional) Save the raw reward data to its own CSV
df_summary = df[["training_iteration"] + reward_cols]
df_summary.to_csv("reward_summary.csv", index=False)

# 4. Plot but save to a PNG file instead of showing
plt.figure(figsize=(8,5))
plt.plot(df["training_iteration"], df["env_runners/episode_reward_mean"], label="episode_reward_mean")

# If you want each policy’s reward-mean too:
for col in reward_cols:
    if "policy" in col and "mean" in col:
        plt.plot(df["training_iteration"], df[col], label=col)

plt.xlabel("Training Iteration")
plt.ylabel("Reward")
plt.title("Reward Curves")
plt.legend(loc="best")
plt.grid(True)

# Save the figure
plt.savefig("reward_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("→ Saved reward_summary.csv and reward_curve.png")
