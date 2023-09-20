from collections import deque
import sys
import math
import numpy as np


def interact(env, agent, num_episodes=20000, window=100):
    """
    It is a agent's peformance

    Parameters:
    env:OpenAI Gym's Taaxi-v1 environment
    agent:Rl agent
    num_episodes:number of episodes of agent-environment
    window:number of episodes to consider when calculating average rewards


    avg_rewards:deque contains the average rewards
    best_avg_rewards:largest value of avg_rewards deque

    """

    avg_rewards = deque(maxlen=num_episodes)
    best_avg_rewards = -math.inf

    samp_rewards = deque(maxlen=window)

    for episode in range(1, num_episodes + 1):
        state = env.reset()

        samp_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            samp_reward += reward
            state = next_state

            if done:
                samp_rewards.append(samp_reward)
                break

            if episode >= 100:
                avg_reward = np.mean(samp_rewards)
                avg_rewards.append(avg_reward)
