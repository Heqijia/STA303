# Abstract Agent Class
class BaseAgent:
    def __init__(self, env, hyperparams):
        """
        Base agent class with shared attributes and methods
        """
        self.env = env
        self.num_actions = env.action_space.n
        self.state_dim = env.observation_space.n
        self.hyperparams = hyperparams

    def select_action(self, state):
        raise NotImplementedError

    def learn(self, batch_size, done):
        raise NotImplementedError

    def update_epsilon(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError