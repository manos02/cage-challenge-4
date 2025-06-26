import glob, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import time

def plot(dir_path):
    
    # Load the csv files into a list of 1d array
    csv_files = sorted(glob.glob(f"{dir_path}/*.csv"))
    
    returns = []
    data_list = []
    meta_info_list = []

    for f in csv_files:
        df = pd.read_csv(f)
        if len(df.columns) > 100:
            returns.append(df.iloc[:, 2].values)
        else:
            returns.append(df.iloc[:, 1].values)

        data_list.append(returns)

        meta_info_list.append({
            "lr": df['lr'].iloc[0] if 'lr' in df.columns else None,
            "clip_param": df['clip_param'].iloc[0] if 'clip_param' in df.columns else None,
            "train_batch_size": df['train_batch_size'].iloc[0] if 'train_batch_size' in df.columns else None,
            "minibatch_size": df['minibatch_size'].iloc[0] if 'minibatch_size' in df.columns else None,
        })
    
    labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    # Get the algorithm names
    # Filenames have to be in the form PPO_trial1.csv for example
    algorithm_names = [label.split("_")[0] for label in labels]

    # Trim to the shorted length
    min_len = max(map(len, returns))
    returns = [d[:min_len] for d in returns]

    episodes = np.arange(min_len)
    rows = []
    for algorithm, trial_name, data in zip(algorithm_names, labels, returns):
        for ep, val in zip(episodes, data):
            rows.append((ep, algorithm, trial_name, val))
    df_long = pd.DataFrame(rows, columns=["episode", "algorithm", "trial", "return"])

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

    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Average return', fontsize=16)

    # Create hyperparameter summary string
    meta_text_lines = []
    for label, meta in zip(labels, meta_info_list):
        meta_text_lines.append(
            f"{label}: lr={meta['lr']}, clip={meta['clip_param']}, "
            f"batch={meta['train_batch_size']}, minibatch={meta['minibatch_size']}"
        )

    meta_text = "\n".join(meta_text_lines)

    # Place the hyperparameter box below the plot
    plt.gcf().text(
        0.55, -0.08, meta_text,
        fontsize=10,
        ha='center',
        va='top',
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.7)
    )


    plt.tight_layout()
    plt.savefig("results/ippo_hyperparameter/ippo_hyper_5.png", bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input directory for plotting")
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dir_path = args.path
    if os.path.isdir(dir_path):
        plot(args.path)
    else:
        print("Invalid directory")



