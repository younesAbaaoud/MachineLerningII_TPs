import gymnasium as gym 
import numpy as np

env = gym.make("FrozenLake-v1",is_slippery=False)




# for _ in range(20):
#      action = env.action_space.sample()
#      print(action)
#      observation , reward , done ,_ ,_ = env.step(action)
#      print(f"Action : {action} , Observation : {observation} , Reward  : {reward}")
     
#      if done :
#          env.reset()

# print(f"Espace d'action : {env.action_space}")
# print(f"Esoace d'observation : {env.observation_space}")

nbr_etat = env.observation_space.n
nbr_action = env.action_space.n

q_table = np.zeros((nbr_etat,nbr_action))

# print("\nQ-Table initiale :")
# print(q_table)

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01 
num_episodes = 5000

for episode in range(num_episodes):
     state, _ = env.reset()
     
     done = False
     while not done:
          if np.random.rand() < epsilon:
            action = env.action_space.sample()  
          else:
            action = np.argmax(q_table[state, :])  
            
          next_state, reward, done, _, _ = env.step(action)
          q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
          state = next_state
     epsilon = max(min_epsilon, epsilon * epsilon_decay)
     
env.close()
print(q_table)

num_test_episodes = 100
successes = 0
for _ in range(num_test_episodes):
     state, _ = env.reset()
     done = False
     while not done:
      action = np.argmax(q_table[state, :])  
      next_state, reward, done, _, _ = env.step(action)
      if reward == 1 :
          successes += 1
      state = next_state
print(f"Taux de réussite après entraînement : {successes}/{num_test_episodes} ({(successes/num_test_episodes) * 100:.2f}%)")

          

     