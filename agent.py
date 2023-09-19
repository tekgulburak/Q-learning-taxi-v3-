import numpy as np
from collections import defaultdict


class Agent:
    def __init__(self, num_act=6):
        """
        Parameters:

        num_act:number of actions
        There can be six actions for actions.These are right,left,up,down,pick up and pick off actions.

        """
        self.num_act = num_act
        self.Q = defaultdict(lambda: np.zeros(self.num_act))
        self.epsilon = 1
        self.gamma = 0.9
        self.alpha = 0.2
        self.episode = 0
        print(
            "gamma:" + str(self.gamma),
            "\n",
            "epsilon:" + self.epsilon,
            "\n",
            "alpha:" + str(self.alpha),
        )

    def select_action(self, state):
        # given the state and agent select the action

        action_probability = np.ones(self.num_act) * self.epsilon / self.num_act

        action_probability[np.argmax(self.Q[state])] += 1 - self.epsilon

        return np.random.choice(self.num_act, action_probability)

    def greedy_action(self, state):
        return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        # update the agents knowledge
        target_reward = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target_reward - self.Q[state][action])
        if done:
            self.episode += 1
            self.epsilon = 1 / self.epsilon
