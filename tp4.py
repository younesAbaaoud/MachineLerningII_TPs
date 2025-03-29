import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3")
state_size = env.observation_space.n
action_size = env.action_space.n

policy_table = np.ones((state_size, action_size)) / action_size  # Politique uniformément répartie
value_table = np.zeros(state_size)

print("Policy Table (premières lignes) :")
print(policy_table[:5])
print("Value Table (premières lignes) :")
print(value_table[:5])


num_episodes = 20

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    print(f"Épisode {episode+1} :")
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        print(f"Action : {action}, Récompense : {reward}")
        state = next_state
    print("-" * 40)

gamma = 0.99
lr_policy = 0.1
clip_epsilon = 0.2

episode_states = []   # Liste des états visités durant l'épisode
episode_actions = []  # Liste des actions effectuées
episode_rewards = []  # Liste des récompenses obtenues

for t in range(len(episode_states)):
    R_t = sum([gamma**i * r for i, r in enumerate(episode_rewards[t:])])
    advantage = R_t - value_table[episode_states[t]]
    r_theta = 0
    # Fonction objectif PPO
    L = min(r_theta * advantage, np.clip(r_theta, 1 - clip_epsilon, 1 + clip_epsilon) * advantage)

num_eval_episodes = 5
total_rewards = []
max_steps = 200

for ep in range(num_eval_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0
    print(f"Épisode {ep+1}:")
    while not done and step < max_steps:
        # Sélectionner l'action optimale selon la table de politique après entraînement
        action = np.argmax(policy_table[state])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Affichage de l'action et de la récompense pour cette étape
        print(f"Étape {step}: Action: {action}, Récompense: {reward}")

        state = next_state
        step += 1

    if step >= max_steps:
        print("Limite d'étapes atteinte.")

    total_rewards.append(total_reward)
    print(f"Récompense cumulée de l'épisode {ep+1}: {total_reward}\n")

print("Récompenses cumulées sur les épisodes d'évaluation :", total_rewards)