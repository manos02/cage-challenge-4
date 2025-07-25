
from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from EnterpriseMAEHmarl import EnterpriseMAE
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPO
from hmarl_action_mask_model import TorchActionMaskModelHppo
import ray
from ray.train import RunConfig
from ray.tune import Tuner, TuneConfig
from helper import parse_args
import gymnasium

# from ray.rllib.utils.annotations import override

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Based on https://github.com/adityavs14/Hierarchical-MARL/blob/main/h-marl-3policy/subpolicies/train_subpolicies.py

ModelCatalog.register_custom_model(
    "hmarl_model", TorchActionMaskModelHppo
)

# Number of blue agents and mapping to policy IDs
NUM_AGENTS = 5
POLICY_MAP = {}

for i in range(NUM_AGENTS):
    POLICY_MAP[f"blue_agent_{i}_master"]  = f"Agent{i}_master"
    POLICY_MAP[f"blue_agent_{i}_investigate"]  = f"Agent{i}_investigate"
    POLICY_MAP[f"blue_agent_{i}_recover"]  = f"Agent{i}_recover"



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


    OBSERVATION_SPACE = {}
    ACTION_SPACE = {}

    for i in range(NUM_AGENTS):
        OBSERVATION_SPACE[f"Agent{i}_master"] = gymnasium.spaces.Dict({'action_mask': gymnasium.spaces.multi_discrete.MultiDiscrete([2,2]),'observations':env.observation_space(f'blue_agent_{i}')['observations'], 'id':gymnasium.spaces.discrete.Discrete(1)})
        ACTION_SPACE[f"Agent{i}_master"] = gymnasium.spaces.discrete.Discrete(2)

        OBSERVATION_SPACE[f"Agent{i}_investigate"] = gymnasium.spaces.Dict({'action_mask': env.observation_space(f"blue_agent_{i}")['action_mask'], 'observations':env.observation_space(f"blue_agent_{i}")['obs_investigate']})
        ACTION_SPACE[f"Agent{i}_investigate"] = env.action_space(f"blue_agent_{i}")

        OBSERVATION_SPACE[f"Agent{i}_recover"] = gymnasium.spaces.Dict({'action_mask': env.observation_space(f"blue_agent_{i}")['action_mask'], 'observations':env.observation_space(f"blue_agent_{i}")['obs_recover']})
        ACTION_SPACE[f"Agent{i}_recover"] = env.action_space(f"blue_agent_{i}")


    config = (
        PPOConfig()
        .framework("torch")
        .debugging(log_level='DEBUG') 
        .environment(
            env="CC4",
        )
        .resources(
            num_gpus=1, # Use if GPUs are available
        )
        .env_runners(
            batch_mode="complete_episodes",
            num_env_runners=31, # parallel sampling, set 0 for debugging
            num_cpus_per_env_runner=1,
            sample_timeout_s=None, # time for each worker to sample timesteps
        )
        .multi_agent(
            policies={
                ray_agent: PolicySpec(
                    policy_class = PPOTorchPolicy,
                    observation_space = OBSERVATION_SPACE[ray_agent],
                    action_space = ACTION_SPACE[ray_agent],
                    config = {"entropy_coeff": 0.001},
                )
                for ray_agent in OBSERVATION_SPACE
            },
            policy_mapping_fn=policy_mapper,
        )
        .training(
            model={"custom_model": "hmarl_model"},
            lr=1e-5,
            grad_clip_by="global_norm",
            grad_clip=0.2,
            train_batch_size=150000, 
            minibatch_size=6000
        )
        .experimental(
            _disable_preprocessor_api=True,  
        )
    )

    return config

def run_training(cluster):
    if cluster and not ray.is_initialized():
        # Connect to the cluster
        ray.init(address="auto")


    config = build_algo_config()

    tuner = Tuner(
        PPO,                              
        param_space=config,
        tune_config=TuneConfig(
            num_samples=1, # how many Optuna trials. Each time with different sampling 
        ),
        run_config=RunConfig(
            storage_path="~/projects/cage-challenge-4/Hmarl/ray_results",
            stop={"training_iteration": 200},
        ),
    )

    result_grid = tuner.fit()


if __name__ == "__main__":
    
    args = parse_args()     
    run_training(args.cluster)
