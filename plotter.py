import glob, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the csv files into a list of 1d array
csv_files = sorted(glob.glob("baseline_results/*.csv"))
data_list = [pd.read_csv(f).iloc[:,2].values for f in csv_files]
labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

# Get the algorithm names
# Filenames have to be in the form PPO_trial1.csv for example
algorithm_names = [label.split("_")[0] for label in labels]

# Trim to the shorted length
min_len = min(map(len, data_list))
data_list = [d[:min_len] for d in data_list]

episodes = np.arange(min_len)
rows = []
for algorithm, trial_name, data in zip(algorithm_names, labels, data_list):
    for ep, val in zip(episodes, data):
        rows.append((ep, algorithm, trial_name, val))
df_long = pd.DataFrame(rows, columns=["episode", "algorithm", "trial","return"])


# Plot with a 95% credible interval around the mean
sns.set_style("whitegrid", {'axes.edgecolor':'black'})
plt.figure(figsize=(8,5))
sns.lineplot(
    x="episode",
    y="return",
    hue="algorithm", # grouping variable
    data=df_long,
    errorbar=('ci', 95), # draws the shaded 95%‐CI around the mean
    lw=2,            # line‐width
)

plt.xlabel('Training Episodes $(\\times10^5)$', fontsize=16)
plt.ylabel('Average return', fontsize=16)
plt.tight_layout()
plt.savefig("myplot.png")
