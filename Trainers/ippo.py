from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG import CybORG
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from CybORG.Agents.IPPO.ippo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from pprint import pprint




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
        self.episode_len = 500
        self.init_env()
    
    def init_env(self):
        # Set up the CybORG environment and PPO agents
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.episode_len
        )

        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add seed for reproducibility
        env = BlueFlatWrapper(env=cyborg)  # Wrap environment for flattened blue agent observation/action space
        env.reset()  # Reset environment to initial state
        self.env = env

        
        # Initialize PPO agents for each blue agent (total 5 agents)
        self.agents = {
            f"blue_agent_{agent}": PPO(
                env.observation_space(f'blue_agent_{agent}').shape[0],
                len(env.get_action_space(f'blue_agent_{agent}')['actions']),
                self.max_eps * self.EPISODE_LENGTH,  # Total training steps
                agent,
                self.messages  # Use message passing if enabled
            )
            for agent in range(5)
        }
        print(f'Using agents {self.agents}')   
        

    def run(self):


        # register_env(name="CC4", env_creator=lambda config: self.env_creator_CC4(config))
        # env = self.env_creator_CC4({})

        # algo_config = (
        #     PPOConfig()
        #     .framework("torch")
        #     .environment(env="CC4")
        #     .debugging(logger_config={"logdir":"logs/train_marl", "type":"ray.tune.logger.TBXLogger"})
        #     .rollouts(
        #         batch_mode="complete_episodes",
        #         num_rollout_workers=30, # for debugging, set this to 0 to run in the main thread
        #     )
        #     .training(
        #         # model={"custom_model": "my_model"},
        #         # sgd_minibatch_size=32768, # default 128
        #         # train_batch_size=1000000, # default 4000
        #         gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256,
        #     )
        #     .multi_agent(policies={
        #         ray_agent: PolicySpec(
        #             policy_class=None,
        #             observation_space=env.observation_space(cyborg_agent),
        #             action_space=env.action_space(cyborg_agent),
        #             config={"gamma": 0.85},
        #         ) for cyborg_agent, ray_agent in self.policy_map.items()
        #     },
        #     policy_mapping_fn=self.policy_mapper
        # ))

        algo = algo_config.build()

        for i in range(50):
            train_info=algo.train()
            pprint(train_info)

        algo.save("results")
                        





