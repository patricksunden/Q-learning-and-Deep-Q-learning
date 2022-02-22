import gym
import numpy as np
import random
import time
import ray
from ray import tune
import time
import math
import pandas as pd 
import matplotlib.pyplot as plt
from collections import deque



env = gym.make("CartPole-v0")
print(env.action_space.n)

# making the Q-table

buckets = (1,1,6,3)
Q_table = np.zeros(buckets+(env.action_space.n,))


# HYPERPARAMETERS
##################################
discount = 0.99
iterations = 800
min_learning_rate = 0.001
min_epsilon = 0.001

upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]
        

epi_100_reward = deque(maxlen=100)
total = 0
total_reward = 0
prior_reward = 0
plot_y = []
plot_x = []
plot_yy = []
plot_xx = []
plot_x_hyper = []
plot_y_hyper = []
plot_lr_hyper = []
plot_cumu_y = []
plot_cumu_x = []
cumu_reward = 0
plot_y_epi = []
plot_x_epi = []
avg_100 = 0


def plot_per_episode(plot_x, plot_y):

	plt.plot(plot_x, plot_y)
	plt.title('CartPole-v0 rewards' )
	plt.ylabel('Rewards')
	plt.xlabel('Iterations')
	plt.axhline(y = 195, color = 'r', linestyle = '-')
	
	plt.show()

def plot_hyper(plot_x_hyper, plot_y_hyper):

	plt.plot(plot_x_hyper, plot_y_hyper)
	plt.title('Epsilon/iterations')
	plt.ylabel('Epsilon')
	plt.xlabel('Iterations')
	plt.show()

def plot(plot_x, plot_y):

	plt.plot(plot_x, plot_y)
	plt.title('CartPole-v0 average 100 episode rewards' )
	plt.ylabel('Rewards')
	plt.xlabel('Iterations')
	plt.axhline(y = 195, color = 'r', linestyle = '-')
	
	plt.show()


def plot_cumu(plot_x, plot_y):

	plt.plot(plot_x, plot_y)
	plt.title('CartPole-v0 Q-learning cumulative rewards' )
	plt.ylabel('Rewards')
	plt.xlabel('Iterations')
	
	
	plt.show()



def epsilon_action(state_value, epsilon):
	if np.random.random() < epsilon:
		#take the best action according to the q table
		action = np.random.randint(0, env.action_space.n)

	else:
		action = np.argmax(Q_table[discrete_state])

	return action
		#do exploration = either move 0 or 1
		
#epsilon update and lr update equation adapted fromhttps://github.com/kevchn/qlearn-cartpole/blob/master/src/qcartpole-methods.py
def epsilon_update(x):
	return max(min_epsilon, min(1.0, 1.0 - math.log10((x+1)/5)))


def lr_update(x):
	return max(min_learning_rate, min(1.0, 1.0 - math.log10((x+1)/5)))


##################################
def get_discrete_state(state):

	discretized = list()
	for i in range(len(state)):
		scaling = ((state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]))
		new_obs = int(round((buckets[i] -1) * scaling))
		new_obs = min(buckets[i] - 1, max(0, new_obs))
		discretized.append(new_obs)

	return tuple(discretized)	

if __name__ == '__main__':

	# Q-learning algorithm
	##################################
	clock = time.time()

	k = 0
	for i in range(iterations + 1):
		
		epsilon = epsilon_update(i)
		learning_rate = lr_update(i)

		discrete_state = get_discrete_state(env.reset())
		done = False
		episode_reward = 0

		while not done:

			action = epsilon_action(discrete_state, epsilon)
			
			new_state, reward, done, _ = env.step(action)

			

			episode_reward += reward
			new_discrete_state = get_discrete_state(new_state)
			max_future_q = np.max(Q_table[new_discrete_state])
			current_q = Q_table[discrete_state + (action,)]
			Q_table[discrete_state + (action,)] += (learning_rate * (reward + discount * max_future_q - current_q))
			discrete_state = new_discrete_state
			
			#if i % 100 == 0:
				#env.render()
			if k == 0 and episode_reward >= 195:
				print("195 points reached at ", i)
				k+=1
			plot_x_hyper.append(i)
			plot_y_hyper.append(epsilon)
			plot_lr_hyper.append(learning_rate)


		cumu_reward += episode_reward
		plot_cumu_y.append(cumu_reward)
		plot_cumu_x.append(i)
		total_reward += episode_reward #episode total reward
		total += episode_reward
		prior_reward = episode_reward
		
		if i % 1 == 0:
			plot_y_epi.append(episode_reward)
			plot_x_epi.append(i)
			epi_100_reward.append(episode_reward)
			plot_x.append(i)
			avg_100 = np.mean(epi_100_reward)
			
		plot_y.append(np.mean(epi_100_reward))

		#if i == 100:
			#plot(plot_x_epi, plot_y_epi)
		

		if i % 100 == 0: #every 100 episodes print the average time and the average reward
			

			mean_reward = total_reward / 100

			plot_yy.append(mean_reward)
			plot_xx.append(i)

			print("Mean Reward, iteration: " + str(mean_reward), i)
			print("epsilon: ", epsilon)

		if avg_100 > 195 and k == 1:
			print("total average reward: ", (total/i))
			print("Congratulations! you have passed the cartpole test in: !", i)
			print("epsilon: ", epsilon)
			k+=1
			
				

				
			total_reward = 0
	clock_end = time.time()		
	print(clock_end)
	print(clock_end - clock)		
	plot(plot_x, plot_y)
	plot(plot_xx, plot_yy)
	plot_hyper(plot_x_hyper, plot_y_hyper)
	plot_hyper(plot_x_hyper, plot_lr_hyper)
	plot_cumu(plot_cumu_x, plot_cumu_y)
	plot_per_episode(plot_x_epi, plot_y_epi)
	
	
	env.close()
	##################################
		
