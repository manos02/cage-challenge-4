
from typing import Dict
from rich import print
from ray.tune import Tuner, TuneConfig
from CybORG import CybORG
from ray.tune.search.optuna import OptunaSearch
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPO
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from mappo_action_mask_model import TorchActionMaskModelMappo
from ray.train import RunConfig
import numpy as np
import ray
from EnterpriseMAEMappo import EnterpriseMAE
from helper import parse_args

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


ModelCatalog.register_custom_model(
    "mappo_model", TorchActionMaskModelMappo
)

# Number of blue agents and mapping to policy IDs
NUM_AGENTS = 5
POLICY_MAP: Dict[str, str] = {
    f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)
}

# Environment creator function
def env_creator_CC4(env_config: dict) -> MultiAgentEnv:
    """
    Instantiate the CybORG Enterprise scenario wrapped for multi-agent.
    """
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    return EnterpriseMAE(env=cyborg)

# Register the environment with RLlib
register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))

# Policy mapping function
def policy_mapper(agent_id, episode, worker, **kwargs) -> str:
    """Map a CybORG agent ID to an RLlib policy ID."""
    return POLICY_MAP[agent_id]


def central_critic_observer(agent_obs, worker, base_env, policies, episode, **kw):
    global_obs = []
    for a in agent_obs.values():
        global_obs.append(a["observations"])
    global_obs = np.concatenate(global_obs, axis=0)
    
    for i in range(5):
        agent_obs[f"blue_agent_{i}"]["global_observations"] = global_obs

    return agent_obs

# Build the PPO algorithm configuration
def build_algo_config():
    """
    Returns a configured PPOConfig for the CC4 multi-agent setup.
    """
    # Instantiate one env to retrieve spaces
    env = env_creator_CC4({})

    # Define multi-agent policies
    policies: Dict[str, PolicySpec] = {
        ray_agent: PolicySpec(
            policy_class=PPOTorchPolicy,
            observation_space=(env.observation_space(cyborg_agent)),
            action_space=env.action_space(cyborg_agent),
            config={
                "entropy_coeff": 0.001
            }
        )
        for cyborg_agent, ray_agent in POLICY_MAP.items()
    }

    
    config = (
        PPOConfig()
        .framework("torch")
        # .debugging(log_level='INFO') 
        .environment(
            env="CC4",
        )
        .resources(
            num_gpus=1,
        )  # Use if GPUs are available
        .env_runners(
            batch_mode="complete_episodes",
            num_env_runners=31, # parallel sampling, set 0 for debugging
            num_cpus_per_env_runner=1,
            sample_timeout_s=None, # time for each worker to sample timesteps
            enable_connectors=False
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapper,
            observation_fn=central_critic_observer,
        )
        .training(
            model={"custom_model": "mappo_model"},
        )
        .experimental(
            _disable_preprocessor_api=True,  
        )     
    )
    return config


def optuna_space(trial):
    """ 
    trial.suggest samples one value from the interval provided
    """
    return {
        "training": {
            "lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True), 
            "clip_param": trial.suggest_float("clip_param", 0.1, 0.3),
            "train_batch_size": trial.suggest_int("train_batch_size", 100000, 200000), 
            "minibatch_size":   trial.suggest_int("minibatch_size", 2048, 8192, log=True),
        },
    }

def run_training():

    if CLUSTER:
        # Connect to the cluster
        ray.init(address="auto")

    """
    Build and run the PPO algorithm for a fixed number of iterations, saving models.
    """
    optuna_search = OptunaSearch(
        metric="env_runners/episode_reward_mean", # Training result objective
        mode="max", # Maximize the objective
        space=optuna_space,
    )

    # Performs early stopping for underperforming trials. Optimizes episode_rewarid_mean
    asha = ASHAScheduler(
        metric="env_runners/episode_reward_mean",
        mode="max",
        max_t=50, # trials that survive long enough get stopped at 50 iters
        grace_period=5, # stop a trial if it is longer than 5 iterations
    )
    
    config = build_algo_config()
    tuner = Tuner(
        PPO,                              
        param_space=config.to_dict(),
        tune_config=TuneConfig(
            search_alg=optuna_search,
            scheduler=asha,
            num_samples=5, # how many Optuna trials. Each time with different sampling 
        ),
        run_config=RunConfig(
            storage_path="~/projects/cage-challenge-4/Mappo/ray_results",
        )
    )

    result_grid = tuner.fit()

    best_res = result_grid.get_best_result(
        metric="env_runners/episode_reward_mean",
        mode="max",
    )
    print("Best config:", best_res.config)
    print("Best res metrics:", best_res.metrics)

if __name__ == "__main__":
    
    CLUSTER = parse_args()
    run_training()

"""
1. Num_env_runners creates x copies of the environment
2. Each runner samples episodes to completion, since batch_mode = 'complete_episodes"
3. The big batch is shuffled into mini-batches
4. For a number of epochs, it iterates over each mini batch and performs:
    - Policy loss
    - Value loss
    - Entropy bonus
5. Each minibatch update moves the network weights
6. ASHA examines the episode reward mean
"""