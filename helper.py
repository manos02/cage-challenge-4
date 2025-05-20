import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process input parameters for agent training")

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster",              
        action="store_true", 
        help="Run under SLURM (submit via sbatch)."
    )
    args = parser.parse_args()
    cluster = args.cluster

    return cluster
    