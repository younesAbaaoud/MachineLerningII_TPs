from traffic_env import TrafficEnvironment
import numpy as np
import matplotlib.pyplot as plt

env = TrafficEnvironment()
state = env.reset()

for _ in range(10):
    action = 0
    next_state , reward = env.step(action)
    print(f"Etat : {next_state}, Recompense : {reward}")

q_table = np.zeros((10, 10, 10, 10, 2))
print(q_table)


def train_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.995, max_steps=1000):
    num_states = 10
    num_actions = 2

    q_table = np.zeros((num_states, num_actions))
    rewards_list = []

    for episode in range(episodes):
        print(f"\nÉpisode {episode + 1}/{episodes}")
        state = env.reset()

        if isinstance(state, (list, np.ndarray)):
            state = state[0]
        state = int(state)

        done = False
        step_count = 0
        total_reward = 0

        while not done and step_count < max_steps:
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(q_table[state])

            step_result = env.step(action)

            if len(step_result) == 2:
                next_state, reward = step_result
                done = (step_count >= max_steps - 1)
            else:
                next_state, reward, done, _ = step_result

            if isinstance(next_state, (list, np.ndarray)):
                next_state = next_state[0]

            next_state = np.clip(int(next_state), 0, num_states - 1)

            best_next_action = np.argmax(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

            total_reward += reward
            state = next_state
            step_count += 1

        rewards_list.append(total_reward)
        epsilon *= decay

    return q_table, rewards_list

q_table, rewards = train_q_learning(env)



import numpy as np

def train_sarsa(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.995, max_steps=1000):
    num_states = 10
    num_actions = 2

    q_table = np.zeros((num_states, num_actions))
    rewards_list = []

    for episode in range(episodes):
        print(f"\nÉpisode {episode + 1}/{episodes}")
        state = env.reset()

        if isinstance(state, (list, np.ndarray)):
            state = state[0]
        state = int(state)

        done = False
        step_count = 0
        total_reward = 0

        action = np.random.choice(num_actions) if np.random.rand() < epsilon else np.argmax(q_table[state])

        while not done and step_count < max_steps:
            step_result = env.step(action)

            if len(step_result) == 2:
                next_state, reward = step_result
                done = (step_count >= max_steps - 1)
            else:
                next_state, reward, done, _ = step_result

            if isinstance(next_state, (list, np.ndarray)):
                next_state = next_state[0]

            next_state = np.clip(int(next_state), 0, num_states - 1)

            next_action = np.random.choice(num_actions) if np.random.rand() < epsilon else np.argmax(q_table[next_state])

            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

            total_reward += reward
            state, action = next_state, next_action
            step_count += 1

        rewards_list.append(total_reward)
        epsilon *= decay

    return q_table, rewards_list

q_table_sarsa, rewards_sarsa = train_sarsa(env)




plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Q-learning", color='b')
plt.plot(rewards_sarsa, label="SARSA", color='r')
plt.xlabel('Épisodes')
plt.ylabel('Récompense cumulée')
plt.title('Comparaison des récompenses: Q-learning vs SARSA')
plt.legend()
plt.show()


