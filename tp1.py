import gymnasium as gym

env = gym.make("CartPole-v1",render_mode="human")
# env.reset()
# print(f"Espace d'action : {env.action_space}")
# print(f"Esoace d'observation : {env.observation_space}")
# l =[]
# for _ in range(100):
#     action = env.action_space.sample()
#     print(action)
#     observation , reward , done ,_ ,_ = env.step(action)
#     print(f"Action : {action} , Observation : {observation} , Reward  : {reward}")
#     l.append((observation,reward,done))
#     if done :
#         env.reset()
# env.close()



a = int(input("enter une action : (0/1"))

env.reset()
print(f"Espace d'action : {env.action_space}")
print(f"Esoace d'observation : {env.observation_space}")

for _ in range(100):

    print(a)
    observation , reward , done ,_ ,_ = env.step(a)
    print(f"Action : {a} , Observation : {observation} , Reward  : {reward}")

    if done :
        env.reset()
env.close()
print(observation,reward,done)

