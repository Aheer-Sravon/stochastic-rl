from stochastic_rl.distributions.knapsack_items import sample_knapsack_item

class StochasticKnapsackEnv:
    """
    Online Stochastic Knapsack Environment (finite horizon)

    State: (c, t, w, v)
    Action: 0 = reject, 1 = accept
    Reward: v if accepted else 0
    Episode ends when t > T or capacity == 0
    """

    def __init__(self, C, T, w_max, seed=None):
        self.C = C
        self.T = T
        self.w_max = w_max
        self.seed = seed

        self.reset()

    def reset(self):
        """Start a new episode"""
        self.c = self.C
        self.t = 1
        self.total_value = 0

        self.w, self.v = sample_knapsack_item(self.w_max)
        
        return self._get_state()

    def step(self, action):
        """
        Apply one action and advance the environment.

        Returns:
            next_state, reward, done, info_dict
        """

        if action not in (0, 1):
            raise ValueError("Action must be 0 (reject) or 1 (accept).")

        accepted = (action == 1) and (self.w <= self.c)

        if accepted:
            self.c -= self.w
            reward = self.v
            self.total_value += reward
        else:
            reward = 0

        # advance time
        self.t += 1

        # termination
        done = (self.t > self.T) or (self.c == 0)

        if not done:
            self.w, self.v = sample_knapsack_item(self.w_max)

        next_state = self._get_state()

        info = {
            "accepted": accepted,
            "remaining_capacity": self.c,
            "total_value": self.total_value
        }

        return next_state, reward, done, info

    def _get_state(self):
        """Return current state as a tuple (c, t, w, v)."""
        return self.c, self.t, self.w, self.v
