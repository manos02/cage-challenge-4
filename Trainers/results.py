import pandas as pd
import matplotlib.pyplot as plt

# Load the results CSV
df = pd.read_csv("tune_results.csv")

print(df["env_runners/episode_reward_mean"])

# Plot global mean episode reward vs. training iteration
plt.figure(figsize=(8, 5))
plt.plot(
    df["training_iteration"],
    df["env_runners/episode_reward_mean"],
    marker="o",
    linestyle="-",
    label="Global Mean Reward",
)
plt.xlabel("Training Iteration")
plt.ylabel("Mean Episode Reward")
plt.title("Global Reward Curve")
plt.grid(True)
plt.legend()

# SAve 
plt.savefig("global_reward_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved plot to global_reward_curve.png")
