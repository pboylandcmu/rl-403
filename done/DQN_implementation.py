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
			self.learning_rate = 0.00001
		elif(environment_name == 'MountainCar-v0'):
			self.num_actions = 3
			self.state_size = 2
			self.learning_rate = 0.0001
		else:
			print("Invalid environment name\nTry 'CartPole-v0' or 'MountainCar-v0")
			exit(0)
			
		self.model = self.define_model(environment_name)

		self.file_count = 0
		self.model_names = []

		self.file_name = "models-mount2/saved_model"

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

	def q_values(self,state):
		return self.predict(state,self.model)

	def predict(self,state,model=None):
        #put state in a list
		if (model is None):
			model = self.model
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
			name = format(model_file + str(self.file_count) + ".h5")
			self.model_names.append(name)
		else:
			name = model_file + str(self.file_count) + ".h5"
			
		self.file_count += 1
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
		his = self.model.fit(x=np.array(states),y=np.array(outs),epochs=epochs,verbose=verbosity)
		return his.history['loss'][0]

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

	def __init__(self, environment_name, render=False,q_flag=0,eps=0.5,eps_decay=0.000035):

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
			self.epsilon_decay = 0
			self.batch_size = 32
		elif(environment_name == 'MountainCar-v0'):
			self.gamma = 1
			self.epsilon_decay = 0.000035
			self.batch_size = 48
		self.pass_freq = 150
		self.burn_in_memory()

	def lookahead_policy(self,q_values,env):
		best_action = 0
		best_value = float('-inf')
		env = env.unwrapped
		state = env.state
		temp = env.steps_beyond_done
		for a in range(0,2):
			s,reward,done,_ = env.step(a)
			if(done):
				if(reward > best_value):
					best_value = reward
					best_action = a
				continue
			for b in range(0,2):
				s2 ,reward,done,_ = env.step(b)
				if(self.q_flag == 2):
					value1 = self.q_net.q_values(s2)
					value2 = self.q_value_estimator.q_values(s2)
					total = np.add(value1,value2)
					value = np.amax(total)
				else:
					value = self.q_net.q_value(s2)
				if(value > best_value):
					best_value = value
					best_action = a
				env.state = s
				env.done = False
				env.steps_beyond_done = temp
			env.state = state
			env.done = False
			env.steps_beyond_done = temp
		env.state = state
		env.steps_beyond_done = temp
		env.done = False
		return best_action

	def random_policy(self):
		return lambda state : randint(0,self.q_net.num_actions)

	def update_slow_network(self):
		if(self.q_flag != 1):
			return
		#self.q_net.save_model('fast_DQN.h5')
		self.q_value_estimator.model.set_weights(self.q_net.model.get_weights())

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.
		done = False
		state = self.env.reset()

		losses = []

		tot_reward = 0
		while not done:
			if self.q_flag == 2:
				if randint(0,2) == 0:
					temp = self.q_net
					self.q_net = self.q_value_estimator
					self.q_value_estimator = temp
				if(self.epsilon >= np.random.uniform()):
					action = np.random.randint(0,self.q_net.num_actions)
				else:
					q1s = self.q_net.predict(state,self.q_net.model)
					q2s = self.q_value_estimator.predict(state,self.q_value_estimator.model)
					action = np.argmax([q1+q2 for (q1,q2) in zip(q1s,q2s)])
			else:
				action = self.q_net.epsilon_greedy_action(state,self.epsilon)

			old_state = state
			state, reward, done, _ = self.env.step(action)

			tot_reward += reward
			
			self.replay_memory.append((old_state,action,reward,state,done))
			
			train_on = self.replay_memory.sample_batch(batch_size = self.batch_size)
			
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
				(s1,a,r,_,d) = train_on[i]
				reward = r if d else r + self.gamma * values[i]
				q_pairs.append((s1,a,reward))
			
			losses.append(self.q_net.fit(q_pairs))
			
			#self.env.render()

		#print("choosing time is " + str(choosing_time))
		#print("q_eval time is " + str(q_eval_time))
		#print("fitting time is " + str(fitting_time))
			
		self.epsilon -= self.epsilon_decay
		return tot_reward, losses

	def test(self,episodes,model_file,model_file_2=None,lookahead = False, video = None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.q_net.load_model(model_file)
		if(model_file_2 is not None):
			self.q_value_estimator.load_model(model_file_2)
		total_rewards = []
		self.env.reset()
		if video is not None:
			env = gym.wrappers.Monitor(self.env, video + model_file, force=True)
		else:
			env = self.env
		print(model_file)
		for _ in range(episodes):
			
			state = env.reset()
			done = False
			total_reward = 0
			while not done:
				#self.env.render()
				if(lookahead):
					action = self.lookahead_policy(self.q_net,env)
				elif(self.q_flag == 1 or self.q_flag == 0): 
					action = self.q_net.epsilon_greedy_action(state,0)
				elif(self.q_flag == 2):
					value1 = self.q_net.q_values(state)
					value2 = self.q_value_estimator.q_values(state)
					total_value = np.add(value1,value2)
					action = np.argmax(total_value)
				state, reward, done, _ = env.step(action)
				#self.env.render()
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
	
	def q_b(self,dir,file_base,model_count=66,file_base_2 = None):
		y = []
		x = []
		count = 150
		if(file_base_2 is None and self.q_flag == 2):
			print("q_b not called correctly")
			exit(0)
		for i in range(1,model_count):
			file_name = dir + os.sep + file_base + str(i) + '.h5'
			if(self.q_flag == 2):
				file_name_2 = dir + os.sep + file_base_2 + str(i) + '.h5'
			else:
				file_name_2 = None
			rewards = self.test(20,file_name,model_file_2= file_name_2)
			print(str(150*i + 150) + " episodes : mean = " + str(np.mean(rewards)))
			y.append(np.mean(rewards))
			x.append(count)
			count += 150
			keras.backend.clear_session()
		plt.plot(x,y)
		plt.xlabel("episodes")
		plt.ylabel("average reward")
		if self.q_flag == 2:
			plt.title(self.environment_name + " Double DQN Performance plot")
		else:
			plt.title(self.environment_name + " DQN Performance plot")
		plt.show()

	def q_c(self,dir,file_base,file_base_2 = None):
		if(self.q_flag == 2):
			model_count = 100
			count = 100
			inc = 100
		else:
			model_count = 66
			count = 150
			inc = 150
		y = []
		x = []
		if(file_base_2 is None and self.q_flag == 2):
			print("q_c not called correctly")
			exit(0)
		for i in range(model_count):
			file_name = dir + os.sep + file_base + str(i+1) + '.h5'
			if(self.q_flag == 2):
				file_name_2 = dir + os.sep + file_base_2 + str(i+1) + '.h5'
			else:
				file_name_2 = None
			rewards = self.test(20,file_name,model_file_2= file_name_2,lookahead = True)
			print(str(inc*i + inc) + " episodes : mean = " + str(np.mean(rewards)))
			y.append(np.mean(rewards))
			x.append(count)
			count += inc
			keras.backend.clear_session()
		plt.plot(x,y)
		plt.xlabel("episodes")
		plt.ylabel("average reward")
		if self.q_flag == 2:
			plt.title(self.environment_name + "Two Step Lookahead Double DQN Performance plot")
		else:
			plt.title(self.environment_name + "Two Step Lookahead DQN Performance plot")
		plt.show()

	def q_d(self,dir,file_base,video_dir,model_count=66,file_base_2=None):
		if(file_base_2 is None and self.q_flag == 2):
			print("q_e not called correctly")
			exit(0)
		
		for i in range(0,1):
			print(i*model_count/3)
			file_name = dir + os.sep + file_base + str(1) + '.h5'
			if(self.q_flag == 2):
				file_name_2 = dir + os.sep + file_base_2 + str(1) + '.h5'
			else:
				file_name_2 = None

			self.test(1,file_name,model_file_2 = file_name_2,video=video_dir)


	def q_e(self,dir,file_base,model_count=66,file_base_2=None):
		if(file_base_2 is None and self.q_flag == 2):
			print("q_e not called correctly")
			exit(0)
		file_name = dir + os.sep + file_base + str(model_count-1) + '.h5'
		if(self.q_flag == 2):
			file_name_2 = dir + os.sep + file_base_2 + str(model_count-1) + '.h5'
		else:
			file_name_2 = None
		
		rewards = self.test(100,file_name,model_file_2 = file_name_2)
		mean = np.mean(rewards)
		std = np.std(rewards)
		print("q_e result: mean = ", mean, ", std = ",std)


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--q',dest='qflag',type=int,default=1)
	parser.add_argument('--q_b',dest='q_b',type=int,default=1)
	parser.add_argument('--q_c',dest='q_c',type=int,default=1)
	parser.add_argument('--q_d',dest='q_d',type=int,default=1)
	parser.add_argument('--q_e',dest='q_e',type=int,default=1)
	parser.add_argument('--q_f',dest='q_f',type=int,default=0)

	return parser.parse_args()

def train_single_dqn(dqn,episodes = 10000,save_freq = 150):
	rewards = []
	all_losses = []
	for i in range(episodes):
		print(i)
		reward,losses = dqn.train()
		all_losses.append(losses)
		print("score = ",reward)
		rewards.append(reward)
		print("running average " + str(np.mean(rewards) if len(rewards) < 51 else np.mean(rewards[-50:])))
		if (i + 1) % save_freq == 0:
			dqn.q_net.save_model()
			print("saved model after " + str(i) + " episodes.")
		if (i + 1) % dqn.pass_freq == 0:
			dqn.update_slow_network()
	print("training done")
	return [loss for eploss in all_losses for loss in eploss]

def train_double_dqn(dqn,episodes= 10000,save_freq = 150):
	rewards = []
	all_losses = []
	for i in range(episodes):
		print(i)
		reward,losses = dqn.train()
		all_losses.append(losses)
		print("score = ",reward)
		rewards.append(reward)
		print("running average " + str(np.mean(rewards) if len(rewards) < 51 else np.mean(rewards[-50:])))
		if (i + 1) % save_freq == 0:
			dqn.q_net.save_model(model_file="models-double"+os.sep+"m1")
			dqn.q_value_estimator.save_model(model_file="models-double"+os.sep+"m2")
			print("saved models after " + str(i) + " episodes.")
	print("training done")
	return [loss for eploss in all_losses for loss in eploss]

def main(args):

	args = parse_arguments()
	qflag = args.qflag
	train = args.train
	q_b = args.q_b
	q_c = args.q_c
	q_d = args.q_d
	q_e = args.q_e
	q_f = args.q_f
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	'''
	Stuff referenced in the Readme is below this comment
	'''

	dqn = DQN_Agent('CartPole-v0',q_flag=qflag)
	#dqn = DQN_Agent('MountainCar-v0',q_flag=qflag)
	if(qflag == 1 or qflag == 0):
		if(train):
			losses = train_single_dqn(dqn)
		#these next couple methods have the form
		# dqn.q_(directory_name,saved_model_name)
		if(q_b):
			dqn.q_b('models','saved_model') #Run code for question B single DQN
		if(q_c):
			dqn.q_c('v1ld35','saved_model')
		if(q_d):
			dqn.q_d('v1ld35','saved_model','ZeroVideos/',model_count=66,file_base_2=None) #Run code for question B single DQN 
		if(q_e):
			dqn.q_e('models','saved_model')
		if (q_f):
			plt.plot(range(len(losses)),losses)
			plt.show()
	else:
		if(train):
			losses = train_double_dqn(dqn)
		#these next couple methods have the form
		# dqn.q_(directory_name,saved_model_name,file_base_2 = other saved model name)
		if(q_b):
			dqn.q_b('models-double','m1',file_base_2='m2') # Run code for question B double DQN
		if(q_c):
			dqn.q_c('v2l70d35','m1',file_base_2='m2')
		if(q_d):
			dqn.q_d('v2l70d35','m1','ZeroVideos/',model_count=66,file_base_2='m2') #Run code for question B single DQN 
		if(q_e):
			dqn.q_e('models-double','m1',file_base_2='m2',model_count = 66)
		if (q_f):
			plt.plot(range(len(losses)),losses)
			plt.show()
					
if __name__ == '__main__':
	main(sys.argv)