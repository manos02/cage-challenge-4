
from typing import Dict
from rich import print
from ray.tune import Tuner, TuneConfig
from CybORG import CybORG
from ray.tune.search.optuna import OptunaSearch
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from EnterpriseMAEHmarl import EnterpriseMAE
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPO
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from hmarl_action_mask_model import TorchActionMaskModelHppo
from ray.train import RunConfig
import ray
import numpy as np
from helper import parse_args
from optuna.samplers import TPESampler
import gymnasium

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


ModelCatalog.register_custom_model(
    "hmarl_model", TorchActionMaskModelHppo
)

class CCPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        self.config = config
        # just in case we are interested, the policy if is in self.config["__policy_id"]


    def handle_extra_ticks(self, postprocessed_batch):
        rewards = None

        # shifting master rewards by -1
        if "id" not in postprocessed_batch["obs"]:
            return postprocessed_batch

        if "rewards" not in postprocessed_batch:
            return postprocessed_batch

        if len(postprocessed_batch["rewards"]) <= 1:
            return postprocessed_batch

        rewards = postprocessed_batch["rewards"][1:]
        rewards = np.concatenate((rewards,[0]))
        postprocessed_batch["rewards"] = rewards

        return postprocessed_batch


    # @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # handle extra ticks first, update rewards
        sample_batch = self.handle_extra_ticks(sample_batch)

        # continue with the default postprocessing (i.e., computing advantages)
        return super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
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
                    # The CCPPOTorchPolicy class correctly matches the reward for the master.
                    # This is not necessary because we only need the subpolicies for H-MARL Expert, 
                    # while the master follows an encoded rule.
                    # Furthermore, we encountered errors when loading the models trained with CCPPOTorchPolicy 
                    # on another rllib instalation and the fix was 
                    # to restore and save the models during evaluation (see evaluation script)

                    # policy_class = CCPPOTorchPolicy,
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
            train_batch_size=100000,
            minibatch_size=4000
        )
        .experimental(
            _disable_preprocessor_api=True,  
        )
    )

    config = config.build()
    return config



if __name__ == "__main__":
    
    CLUSTER = parse_args()  
    algo = build_algo_config()

    if CLUSTER:
        # Connect to the cluster
        ray.init(address="auto")

    for i in range(50):
        iteration = i # for restore, adjust iter, overwise you will  overwrite old models, e.g.  i + 156
        train_info = algo.train()
        print("\nIteration:", i, train_info)
        # model_dir_crt = os.path.join(model_dir, "iter_"+str(iteration))
        # print("\nSaving model in:", model_dir_crt)
        # algo.save(model_dir_crt)

    algo.save("results")