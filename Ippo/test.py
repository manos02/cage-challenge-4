
from typing import Dict
from rich import print
from ray.tune import Tuner, TuneConfig
from CybORG import CybORG
from ray.tune.search.optuna import OptunaSearch
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPO
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ippo_action_mask_model import TorchActionMaskModelIppo
from ray.train import RunConfig
import ray
import torch
import os
import numpy as np
from cc4_env import TorchWrapper
from torchrl.envs import GymWrapper

# Torch
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm


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
    return TorchWrapper(env=cyborg, num_env=env_config["num_env"], device=env_config["device"])
    # return GymWrapper(cyborg)


if __name__ == "__main__":
    
    # Devices
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    
    num_envs = 10

    env_config = {
        "num_env":num_envs,
        "device":device
    }
    
    env = env_creator_CC4(env_config=env_config)
    obs, _ = env.reset()

    actions = {'blue_agent_0': 42}
    messages = {'blue_agent_0': np.array([1, 0, 0, 0, 0, 0, 0, 0])}
    obs, reward, terminated, truncated, info = env.step(actions)    
    print(obs['blue_agent_1'])
    # print(reward['blue_agent_1'])
    # print(reward['blue_agent_2'])
    # print(env.active_agents)
    # print(env.action_space('blue_agent_0'))
    # print(env.action_labels('blue_agent_0'))



    # Sampling
    batch_size = 5000  # Number of team frames collected per training iteration
    n_iters = 50  # Number of sampling and training iterations
    total_steps = batch_size * n_iters

    # Training
    num_epochs = 30  # Number of optimization steps per training iteration
    minibatch_size = 500  # Size of the mini-batches in each optimization step
    lr = 3e-4  # Learning rate
    max_grad_norm = 1.0  # Maximum norm for the gradients

    # PPO
    clip_epsilon = 0.2  # clip value for PPO loss
    gamma = 0.99  # discount factor
    lmbda = 0.9  # lambda for generalised advantage estimation
    entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

    max_steps = 500
    n_agents = 5

    
    # env = TransformedEnv(
    #     env,
    #     RewardSum(in_keys="reward", out_keys=["episode_reward"]),
    # )
    # check_env_specs(env)


   
