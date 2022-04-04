#Code adapted from S. Kumar, Balancing a CartPole System with Reinforcement Learning – A Tutorial arXiv:2006.04938v2 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import random

filename="model_weights_400.h5"
#DENSE is the basic layer form of the meural network

def layers(state_space, action_space, learning_rate):
	model = Sequential()
	model.add(Dense(24, input_dim = state_space, activation ="relu"))# 4-node input layer and a 24 neuron hidden layer
	model.add(Dense(24, activation = "relu"))# 24 neuron hidden layer
	model.add(Dense(action_space, activation = 'linear'))# 2-node output layer
	model.compile(loss = 'mse', optimizer = Adam(learning_rate=learning_rate))
	#model.load_weights("model_weights.h5")
	model.summary()
	return model

class DQN:
	def __init__(self):

		#HYPERPARAMETERS
		#self.weight = 'model_weights.h5'
		self.env = gym.make('CartPole-v0')
		self.env.seed(0)
		self.state_space = self.env.observation_space.shape[0]
		self.action_space = self.env.action_space.n
		print("state_space: ", self.state_space)
		print("action_space", self.action_space)
		self.discount_factor = 0.99
		self.epsilon = 1
		self.epsilon_minimum = 0.01
		self.iterations = 800
		self.learning_rate = 0.001
		self.model = layers(self.state_space, self.action_space, self.learning_rate, self.epsilon, self.epsilon_minimum)
	

		#INFO DATA
		self.max_reward = 500
		self.scores = deque(maxlen = 100)
		self.scores_10 = deque(maxlen = 10) 
		self.mean_score = 0

		#MEMORY
		self.memory = deque(maxlen=2000)
		self.batch_size = 64


	def load(self, name):
		self.model = load_model(name)

	def save(self, name):
		self.model.save(name)

	def epsilon_action(self, state):

		if np.random.rand() <= self.epsilon:
			#take the random action 0 or 1 (left or right)
			return random.randrange(self.action_space)
		else:
			#take best action
			action = self.model.predict(state)
			return np.argmax(action[0])

	def run(self):
		try:
			for iteration in range(1, self.iterations):
				done = False
				index = 0
				state = self.env.reset()
				state = np.reshape(state, [1, self.state_space])
				
				while not done: #done = true if a score of 200 is reached or the pole falls and the run is terminated
					action = self.epsilon_action(state) #The output depends on the epsilon-hyperparameter
					new_state, reward, done, info = self.env.step(action) #Take a step every timestep
					new_state = np.reshape(new_state, [1, self.state_space]) 
					reward = reward if not done else -100 #Give a negative reward if the state is terminal
					self.memory.append((state, action, reward, new_state, done))#Remember state, action ,reward, new state and done for training purposes
					state = new_state
					if self.mean_score < 195 and len(self.memory) > self.batch_size:
						self.training(iteration)

					index += 1# +1 reward for each step that is not terminal/done.
					
				#Goes here when done == true
				print("{} episode, score = {} , epsilon = {} , memory length = {}".format(iteration, index, self.epsilon, len(self.memory)))
				self.scores.append(index)
				self.scores_10.append(index)
				self.mean_score = np.mean(self.scores)
				mean_score_10 = np.mean(self.scores_10)
				print("Last 100 average scores from memory: ", self.mean_score)
				print("Last 10 average scores from memory: ", mean_score_10)		


		finally: #When user presses ctrl+c or max number of iterations reached

				print("-----------Saving Model------------")
				self.save("model_weights_400.h5") #Saves the weights of the neural network, can be used by the test-function
				


	#epsilon update equation adapted from https://github.com/kevchn/qlearn-cartpole/blob/master/src/qcartpole-methods.py
	def epsilon_update(self, x):
		return max(self.epsilon_minimum, min(1.0, 1.0 - math.log10((x+1)/5)))

	def training(self, iteration):

		mini_batch = random.sample(self.memory, self.batch_size)
		current_state = np.zeros((self.batch_size, self.state_space))
		next_state = np.zeros((self.batch_size, self.state_space))
		action, reward, done = [], [], []

		for i in range(self.batch_size):
			current_state[i] = mini_batch[i][0]   # current_state
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			next_state[i] = mini_batch[i][3]  # next_state
			done.append(mini_batch[i][4])
		target = self.model.predict(current_state)
		Qvalue_ns = self.model.predict(next_state)
		
		for i in range(self.batch_size):
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				target[i][action[i]] = reward[i] + self.discount_factor*(np.amax(Qvalue_ns[i]))
		self.model.fit(current_state, target, epochs = 1 , verbose = 0, batch_size=self.batch_size)
		self.epsilon = self.epsilon_update(iteration)

	def test(self):
		self.load("model_weights_400.h5")
		average_reward = 0
		for iteration in range(self.iterations):
			state = self.env.reset()
			state = np.reshape(state, [1, self.state_space])
			done = False
			index = 0
			while not done:
				#self.env.render()
				action = np.argmax(self.model.predict(state))
				next_state, reward, done, _ = self.env.step(action)
				state = np.reshape(next_state, [1, self.state_space])
				index += 1
				if done:
					#
					self.plot_x.append(index) ##tarkista tämä seuraavan kerran kun ajat
					self.plot_y.append(iteration)
					#
					if iteration == (self.iterations-1):
						plot(self.plot_x, self.plot_y)	
						average_results = average_reward/self.iterations
						print("average reward of the test: ", average_results)
					average_reward += index
					print("episode: {}/{}, score: {}".format(iteration, self.iterations, index))
					break

if __name__ == "__main__":
	agent = DQN()
	tf.test.gpu_device_name()
	agent.run()
	#agent.test()
	#weight_reader(filename)
	
