#!/usr/bin/env python
import keras
import tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from numpy.random import randint


class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		if(environment_name == 'CartPole-v0'):
			self.num_actions = 2
			self.state_size = 4
			self.learning_rate = 0.001
		elif(environment_name == 'MountainCar-v0'):
			self.num_actions = 3
			self.state_size = 2
			self.learning_rate = 0.0001
		else:
			print("Invalid environment name\nTry 'CartPole-v0' or 'MountainCar-v0")
			exit(0)
			
		self.model = self.define_model(environment_name)

		self.file_count = 0
		self.file_name = "saved_model"
		self.model_names = []

	def define_model(self,environment_name):
		model = Sequential()
		if(environment_name == 'CartPole-v0'):
			model.add(Dense(16, input_dim=4, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(2, activation='linear'))
		elif(environment_name == 'MountainCar-v0'):
			model.add(Dense(16, input_dim=2, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(16, activation='relu'))
			model.add(Dense(3, activation='linear'))
		model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
              loss='MSE',
              metrics=['accuracy'])
		return model

	def greedy_action(self,state,model=None):
		if(model is None):
			model = self.model
		out = self.predict(state,model)
		return np.argmax(out)

	def q_value(self,state,model = None):
		if(model is None):
			model = self.model
		out = self.predict(state,model)
		return np.amax(out)

	def predict(self,state,model):
        #put state in a list
		s = []
		s.append(state)
		return model.predict(np.array(s))[0]

	def epsilon_greedy_action(self,state,epsilon,model=None):
		if(model is None):
			model = self.model
		if(epsilon >= np.random.uniform()):
			return np.random.randint(0,self.num_actions)
		else: return self.greedy_action(state,model)


	def save_model(self,model_file=None):
		if(model_file is None):
			model_file = self.file_name
		self.file_count += 1
		name = format(model_file + str(self.file_count) + ".h5")
		self.model.save(name)
		self.model_names.append(name)
		return name

	def load_model(self,model_file):
		if(model_file is None):
			return
		self.model =  load_model(model_file)

	def get_model_names(self):
		return self.model_names

	def fit(self,D,epochs=1,verbosity=0):
		states = []
		targets = []
		for (state,action,target) in D:
			states.append(state)
			out = self.predict(state,self.model)
			out[action] = target
			targets.append(out)
		self.model.fit(x=np.array(states),y=np.array(targets),epochs=epochs,verbose=verbosity)
        #score = model.evaluate(states,targets)
        #print(score)
        #return score[1]

	'''
	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights.
		pass

	def save_model_weights(self, suffix):
		#We don't know what to do with this
		pass
	'''

class Replay_Memory():

	def __init__(self, memory_size=50000):
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
		#transition is a 4-tuple of s,a,r,s
		if self.memory is None:
			self.memory = [transition for _ in range(self.memsize)]
		self.memory[self.counter] = transition
		self.counter += 1
		# Appends transition to the memory.

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
		self.env = gym.make(environment_name)
		self.q_net = QNetwork(environment_name)
		self.replay_memory = Replay_Memory() 

		self.epsilon = 0.5
		self.epsilon_decay = 0.000004

		if(environment_name == 'CartPole-v0'):
			self.gamma = 0.99
		elif(environment_name == 'MountainCar-v0'):
			self.gamma = 1

		self.burn_in_memory()

	def epsilon_greedy_policy(self, q_values,epsilon = None):
		if(epsilon is None):
			epsilon = self.epsilon
		return lambda state : q_values.epsilon_greedy_action(state,epsilon)

	def greedy_policy(self, q_values):
		return lambda state : q_values.greedy_action(state)
		# Creating greedy policy for test time.
		pass

	def lookahead_policy(self,q_values,state):
		best_action = 0
		best_value = float('-inf')
		for a in range(0,2):
			s,reward,done = self.env.step(a)
			if(done):
				if(reward > best_value):
					best_value = reward
					best_action = a
				continue
			for b in range(0,2):
				s2 ,reward,done = self.env.step(b)
				value = self.q_net.q_value(s2)
				if(value > best_value):
					best_value = value
					best_action = a
		return best_action

	def random_policy(self):
		return lambda state : randint(0,self.q_net.num_actions)

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.
		done = False
		state = self.env.reset()
		while not done:
			e_greedy = self.epsilon_greedy_policy(self.q_net)
			action = e_greedy(state)
			old_state = state
			state, reward, done, _ = self.env.step(action)

			self.replay_memory.append((old_state,action,reward,state))
			train_on = self.replay_memory.sample_batch()
			q_pairs = [(s1,a,r + self.gamma * (self.q_net.q_value(s2))) for (s1,a,r,s2) in train_on]
			self.q_net.fit(q_pairs)
		self.epsilon -= self.epsilon_decay

		pass

	def test(self,episodes,model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.q_net.load_model(model_file)
		get_action = self.epsilon_greedy_policy(self.q_net,.05)
		total_rewards = []
		for _ in range(episodes):
			state = self.env.reset()
			done = False
			total_reward = 0
			while not done:
				action = get_action(state)
				state, reward, done, _ = self.env.step(action)
				total_reward += reward
			total_rewards.append(total_reward)
		return total_rewards

	def burn_in_memory(self,burn_in=10000):
		done = False
		pol = self.random_policy()
		state = self.env.reset()
		for _ in range(burn_in):
			if done:
				state = self.env.reset()
			action = pol(state)
			old_state = state
			state, reward, done, _ = self.env.step(action)
			self.replay_memory.append((old_state,action,reward,state))

		# Initialize your replay memory with a burn_in number of episodes / transitions.


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

	episodes = 10000
	save_freq = 1500
	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	dqn = DQN_Agent('MountainCar-v0')
	for i in range(episodes):
		dqn.train()
		if i % save_freq == 0:
			dqn.q_net.save_model()
	model_names = dqn.q_net.get_model_names()
	rewards = [dqn.test(model_name) for model_name in model_names]
	
if __name__ == '__main__':
	main(sys.argv)

