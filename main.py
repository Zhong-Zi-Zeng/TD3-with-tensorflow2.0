import gym
import matplotlib.pyplot as plt
from TD3 import TD3


EPISODES = 200

env = gym.make('Pendulum-v0')
agent = TD3(alpha=0.001,
             beta=0.001,
             max_size=50000,
             tau=0.005,
             n_actions=1,
             input_shape = 3,
             min_action_value=-2,
             max_action_value=2)

reward_list = []
for i in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    print('now episode is: {}'.format(i))

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        state = next_state

    reward_list.append(total_reward)

plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.show()