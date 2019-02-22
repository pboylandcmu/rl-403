#!/usr/bin/env python
import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from random import randint


class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		model = self.define_model(environment_name)
		if(environment_name == 'CartPole-v0'):
			self.num_actions = 2
			self.state_size = 4
		elif(environment_name == 'MountainCar-v0'):
			self.num_actions = 3
			self.state_size = 2
		else:
			print("Invalid environment name\nTry 'CartPole-v0' or 'MountainCar-v0")
			exit(0)

	def define_model(self,environment_name):
		model = Sequential()
		if(environment_name == 'CartPole-v0'):
			model.add(Dense(16, input_dim=4, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(2, activation='softmax'))
		elif(environment_name == 'MountainCar-v0'):
			model.add(Dense(16, input_dim=2, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(3, activation='softmax'))
		model.compile(optimizer='Adam',
              loss='MSE',
              metrics=['accuracy'])
		return model

	def greedy_action(self,state,model=self.model):
		return np.argmax(model.predict(state))

	def epsilon_greedy_action(self,state,epsilon,model=self.model):
		if(epsilon >= np.random()):
			return np.randint(0,self.num_actions)
		else: return self.greedy_action(state,model)

	def save_model(self,model_file):
		self.model.save(model_file)

	def load_model(self, model_file):
		return load_model(model_file)

	'''
	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights.
		pass

	def save_model_weights(self, suffix):
		#We don't know what to do with this
		pass
	'''

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):
		self.memory = None
		self.memsize = memory_size
		self.counter = 0
		self.full = False

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		pass

	def sample_batch(self, batch_size=32):
		if self.full:
			return [self.memory[randint(0,self.memsize)] for _ in range(batch_size)]
		return [self.memory[randint(0,self.counter)] for _ in range(batch_size)]
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		# You will feed this to your model to train.
		pass

	def append(self, transition):
		if self.memory is None:
			self.memory = [transition[0] for _ in range(self.memsize)]
		for i in range(len(transition)):
			self.memory[self.counter] = transition[i]
			self.counter += 1
			if self.counter > self.memsize:
				self.counter = 0
				self.full = True
		# Appends transition to the memory.
		pass

class DQN_Agent():

	# In this class, we will implement functions to do the following.
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		pass

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		pass

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		pass

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.
		pass

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		pass

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		pass



def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
	main(sys.argv)

