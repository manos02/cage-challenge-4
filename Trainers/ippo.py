from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG import CybORG
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
# from CybORG.Agents.IPPO.ippo import PPO
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from ray.tune import register_env
from pprint import pprint
from ray.rllib.models import ModelCatalog
from action_mask_model_CC4 import TorchActionMaskModel




# Trainer class for the Proximal Policy Optimization (PPO) algorithm
class PPOTrainer:

    def __init__(self, args):
        # Initialize training settings and configurations
        self.agents = {}  # Dictionary to store PPO agents
        self.total_rewards = []  # List to track cumulative rewards across episodes
        self.average_rewards = []  # List to track average rewards for rollouts
        self.partial_rewards = 0  # Accumulator for rewards in the current rollout
        self.best_reward = -7000  # Best average reward seen so far
        self.count = 0  # Total steps taken across all episodes
        self.load_last_network = args.Load_last  # Flag to load the last saved network
        self.load_best_network = args.Load_best  # Flag to load the best saved network
        self.messages = args.Messages  # Enable or disable message passing between agents
        self.rollout = args.Rollout  # Number of episodes before policy update
        self.max_eps = args.Episodes  # Total number of episodes for training
        self.num_agents = 5
        self.policy_map = {f"blue_agent_{i}": f"Agent{i}" for i in range(self.num_agents)}
        self.ModelCatalog.register_custom_model(
            "my_model", TorchActionMaskModel
        )

    def env_creator_CC4(self, config):
        # Set up the CybORG environment
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500
        )

        cyborg = EnterpriseMAE(CybORG(scenario_generator=sg, seed=1))  # Add seed for reproducibility        
        return cyborg


    def policy_mapper(self, agent_id, episode, worker, **kwargs):
        return self.policy_map[agent_id]


    def run(self):
        register_env(name="CC4", env_creator=lambda config: self.env_creator_CC4(config))
        env = self.env_creator_CC4({})

        algo_config = (
            PPOConfig()
            .framework("torch")
            .environment(env="CC4")
            .debugging(logger_config={"logdir":"logs/train_marl", "type":"ray.tune.logger.TBXLogger"})
            .rollouts(
                batch_mode="complete_episodes",
                num_rollout_workers=30, # for debugging, set this to 0 to run in the main thread
            )
            .training(
                model={"custom_model": "my_model"},
                sgd_minibatch_size=32768, # default 128
                train_batch_size=1000000, # default 4000
            )
            .multi_agent(policies={
                ray_agent: PolicySpec(
                    policy_class=None,
                    observation_space=env.observation_space(cyborg_agent),
                    action_space=env.action_space(cyborg_agent),
                    config={"gamma": 0.85},
                ) for cyborg_agent, ray_agent in self.policy_map.items()
            },
            policy_mapping_fn=self.policy_mapper
        ))

        algo = algo_config.build()

        for i in range(50):
            train_info=algo.train()
            pprint(train_info)

        algo.save("results")
                        





