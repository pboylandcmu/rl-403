#!/usr/bin/env python
import keras
import tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from numpy.random import randint
import time
import os
import matplotlib.pyplot as plt


class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name,qflag = 1):
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
		if(qflag == 0):
			self.file_name = "atari_models/saved_model"
		elif(qflag == 1):
			self.file_name = "models/save_model"
		else:
			self.file_name = "double_models/save_models"
		self.model_names = []

	def define_model(self,environment_name):
		model = Sequential()
		if(environment_name == 'CartPole-v0'):
			model.add(Dense(80, input_dim=4, activation='relu'))
			model.add(Dense(80, activation='relu'))
			model.add(Dense(80, activation='relu'))
			model.add(Dense(2, activation='linear'))
		elif(environment_name == 'MountainCar-v0'):
			model.add(Dense(80, input_dim=2, activation='relu'))
			model.add(Dense(80, activation='relu'))
			model.add(Dense(80, activation='relu'))
			model.add(Dense(3, activation='linear'))
		model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
              loss='MSE')
		return model

	def greedy_action(self,state):
		out = self.predict(state,self.model)
		return np.argmax(out)

	def q_value(self,state):
		out = self.predict(state,self.model)
		return np.amax(out)

	def predict(self,state,model):
        #put state in a list
		s = []
		s.append(state)
		return model.predict(np.array(s))[0]

	def batch_predict_values(self,states):
		predictions = self.model.predict(states)
		return [np.amax(prediction) for prediction in predictions]

	def batch_predict_actions(self,states):
		predictions = self.model.predict(states)
		return [np.argmax(prediction) for prediction in predictions]

	def epsilon_greedy_action(self,state,epsilon,model=None):
		if(epsilon >= np.random.uniform()):
			return np.random.randint(0,self.num_actions)
		else:
			return self.greedy_action(state)


	def save_model(self,model_file=None):
		if(model_file is None):
			model_file = self.file_name
			self.file_count += 1
			name = format(model_file + str(self.file_count) + ".h5")
			self.model_names.append(name)
		else:
			name = model_file
		self.model.save(name)
		return name

	def load_model(self,model_file):
		if(model_file is None):
			return
		self.model = load_model(model_file)

	def get_model_names(self):
		return self.model_names

	def fit(self,D,epochs=1,verbosity=0):
		#D is a bunch of (state,action,target-q)

		states = [state for (state,_,_) in D]
		outs = self.model.predict(np.array(states))
		for i in range(len(D)):
			(_,action,target) = D[i]
			outs[i][action] = target
		self.model.fit(x=np.array(states),y=np.array(outs),epochs=epochs,verbose=verbosity)
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

	def append(self, transition):
		#transition is a 5-tuple of s,a,r,s,d
		if self.memory is None:
			self.memory = [transition for _ in range(self.memsize)]
		self.memory[self.counter] = transition
		self.counter += 1
		if self.counter >= self.memsize:
			self.full = True
			self.counter = 0
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

	def __init__(self, environment_name, render=False,q_flag=0,eps=0.5,eps_decay=0.000025):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.env = gym.make(environment_name)
		self.q_flag = q_flag
		self.q_net = QNetwork(environment_name,qflag = q_flag)
		if(q_flag == 0):
			self.q_value_estimator = self.q_net
		else:
			self.q_value_estimator = QNetwork(environment_name)
		self.replay_memory = Replay_Memory() 

		self.epsilon = eps
		self.epsilon_decay = eps_decay
		self.environment_name = environment_name

		if(environment_name == 'CartPole-v0'):
			self.gamma = 0.99
			self.epsilon_decay = 0.000045
		elif(environment_name == 'MountainCar-v0'):
			self.gamma = 1
			self.epsilon_decay = 0.000045
		self.pass_freq = 150

		self.burn_in_memory()

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

	def update_slow_network(self):
		if(self.q_flag != 1):
			return
		#self.q_net.save_model('fast_DQN.h5')
		self.q_value_estimator.model.set_weights(self.q_net.model.get_weights())
		'''K.clear_session()
		self.q_net.load_model('fast_DQN.h5')
		self.q_value_estimator.load_model('fast_DQN.h5')'''

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.
		done = False
		state = self.env.reset()

		tot_reward = 0
		while not done:
			if self.q_flag == 2:
				if randint(0,2) == 0:
					temp = self.q_net
					self.q_net = self.q_value_estimator
					self.q_value_estimator = temp

			action = self.q_net.epsilon_greedy_action(state,self.epsilon)

			old_state = state
			state, reward, done, _ = self.env.step(action)

			tot_reward += reward
			
			self.replay_memory.append((old_state,action,reward,state,done))
			
			train_on = self.replay_memory.sample_batch()
			
			states = [s for (_,_,_,s,_) in train_on]

			if self.q_flag == 0:
				values = self.q_net.batch_predict_values(np.array(states))
			if self.q_flag == 1:
				values = self.q_value_estimator.batch_predict_values(np.array(states))
			else:
				actions = self.q_net.batch_predict_actions(np.array(states))
				pred_qs = self.q_value_estimator.model.predict(np.array(states))
				values = [qs[a] for (qs,a) in zip(pred_qs,actions)]

			q_pairs = []
			for i in range(len(train_on)):
				(s1,a,r,s2,d) = train_on[i]
				reward = r if d else r + self.gamma * values[i]
				q_pairs.append((s1,a,reward))
			
			self.q_net.fit(q_pairs)
			#self.env.render()

		#print("choosing time is " + str(choosing_time))
		#print("q_eval time is " + str(q_eval_time))
		#print("fitting time is " + str(fitting_time))
			
		self.epsilon -= self.epsilon_decay
		return tot_reward

	def test(self,episodes,lookahead = False,model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.q_net.load_model(model_file)
		total_rewards = []
		for _ in range(episodes):
			state = self.env.reset()
			done = False
			total_reward = 0
			while not done:
				if(lookahead):
					action = self.lookahead_policy(self.q_net,state)
				else: 
					action = self.q_net.epsilon_greedy_action(state,0)
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
			self.replay_memory.append((old_state,action,reward,state,done))

		# Initialize your replay memory with a burn_in number of episodes / transitions.
	
	def q_b(self,dir,file_base,model_count):
		y = []
		x = []
		count = 150
		for i in range(1,model_count+1):
			file_name = dir + os.sep + file_base + str(i) + '.h5'
			rewards = self.test(20,file_name)
			print(str(150*i) + " episodes : mean = " + str(np.mean(rewards)))
			y.append(np.mean(rewards))
			x.append(count)
			count += 150
		plt.plot(x,y)
		plt.xlabel("episodes")
		plt.ylabel("average reward")
		plt.title(self.environment_name + " Performance plot")
		plt.show()


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
	save_freq = 150
	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	dqn = DQN_Agent('CartPole-v0',q_flag=2)
	#dqn = DQN_Agent('MountainCar-v0',q_flag=0)
	#dqn.q_b('models','saved_model',66)
	rewards = []
	for i in range(episodes):
		print(i)
		reward = dqn.train()
		print("score = ",reward)
		rewards.append(reward)
		print("running average " + str(np.mean(rewards) if len(rewards) < 51 else np.mean(rewards[-50:])))
		if (i + 1) % save_freq == 0:
			dqn.q_net.save_model()
			print("saved model after " + str(i) + " episodes.")
		if (i + 1) % dqn.pass_freq == 0:
			dqn.update_slow_network()
	print("training done")

	
if __name__ == '__main__':
	main(sys.argv)

