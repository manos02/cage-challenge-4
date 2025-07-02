
import pandas as pd
import os
import json

def get_return_values(i, dir, parent_path):

    json_path_file = f"{dir}/params.json"
    csv_file_path = f"{dir}/progress.csv"
        
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        df = pd.read_csv(csv_file_path)
        
        # Select only mean return col
        df = df[['env_runners/episode_return_mean']]
        
        if os.path.exists(json_path_file):
            with open(json_path_file, "r") as f:
                data = json.load(f)
                
                # None is the default if a parameter is not found        
                lr = data.get("lr", None) 
                clip_ratio = data.get("clip_param", None) 
                batch_size = data.get("train_batch_size", None) 
                minibatch_size = data.get("minibatch_size", None) 

            # Add hyperparameters as constant columns
            df['lr'] = lr
            df['clip_param'] = clip_ratio
            df['train_batch_size'] = batch_size
            df['minibatch_size'] = minibatch_size

            df.to_csv(os.path.join(parent_path, f'HMARL-{i}_final.csv'))

# Include the desired directories
root_directories = ["Hmarl/ray_results/PPO_2025-07-02_04-47-37"]

for rootdir in root_directories:
    subdirs = [name for name in os.listdir(rootdir)
           if os.path.isdir(os.path.join(rootdir, name))]
    
    for i, dir in (enumerate(subdirs)):
        get_return_values(i, os.path.join(rootdir, dir), rootdir)
        
