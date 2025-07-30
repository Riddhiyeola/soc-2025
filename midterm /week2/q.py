import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)  # simple grid world

# Q-table initialization
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.8        # learning rate
gamma = 0.95       # discount factor
epsilon = 0.1      # exploration rate
episodes = 1000

for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, _, _ = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q-Learning formula
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state

print("Training complete!\n")
