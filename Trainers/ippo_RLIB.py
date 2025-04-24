import os

from statistics import mean, stdev
from typing import Any, Dict, Tuple
from rich import print

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE

from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from .action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Number of blue agents and mapping to policy IDs
NUM_AGENTS = 5
POLICY_MAP: Dict[str, str] = {
    f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)
}

# # Register custom model
# ModelCatalog.register_custom_model(
#     "my_model",
#     TorchActionMaskModel
# )

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
            observation_space=env.observation_space(cyborg_agent),
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
        .debugging(
            logger_config={"logdir": "logs/train_marl", "type": "ray.tune.logger.TBXLogger"}
        )
        .environment(env="CC4")
        # .resources(num_gpus=1)  # Use if GPUs are available
        .experimental(_disable_preprocessor_api=True)
        .env_runners(
            batch_mode="complete_episodes",
            num_env_runners=1,
        )
        .training(
            train_batch_size=128,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapper,
        )
    )
    return config

# Main training entrypoint
def run_training():
    """
    Build and run the PPO algorithm for a fixed number of iterations, saving models.
    """
    model_dir = "models/train_marl"
    algo = build_algo_config().build()

    for i in range(200):
        result = algo.train()
        print(result)
        # print(f"Iteration {i}: reward_mean={result['episode_reward_mean']}")
        # checkpoint_dir = os.path.join(model_dir, f"iter_{i}")
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # algo.save(checkpoint_dir)


if __name__ == "__main__":
    run_training()
