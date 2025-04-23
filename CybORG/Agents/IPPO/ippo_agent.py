from network import ActorCritic
from CybORG.Agents.IPPO.buffer import ReplayBuffer

class PPO:
    def __init__(self, env, obs_space, action_space):
        # Extract environment information
        self.env = env
        self.obs_dim = obs_space
        self.act_dim = action_space

        self.actor = ActorCritic(self.obs_dim, self.act_dim)
        self.critic = ActorCritic(self.obs_dim, 1)


        self._init_hyperparameters()
        
    def learn(self, max_timesteps):
        obs = self.env.reset()
        done = False

        for iteration in range(max_timesteps):
            pass

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch


        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:

            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()
            done = False

