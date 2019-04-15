from __future__ import print_function
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import argparse
import gym
import envs
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import randint
import time
import sys


def concat(list1,list2):
    result = []
    for e in list1:
        result.append(e)
    for e in list2:
        result.append(e)
    return result

class DDPG:
    def __init__(self, env, args):
        # Initialize your class here with relevant arguments
        # e.g. learning rate for actor and critic, update speed
        # for target weights, etc.

        self.render = args.render
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.actor_file = args.actor_file
        self.actor_target_file = args.actor_target_file
        self.critic_file = args.critic_file
        self.critic_target_file = args.critic_target_file
        self.save_dir = args.saved_dir
        self.train_from = args.train_from
        self.actor_file_count = self.train_from
        self.critic_file_count = self.train_from
        self.verbose = args.verbose
        self.tau = args.tau
        self.gamma = args.gamma
        self.env = env
        self.batch_size = 32

        self.actor = self.actor_model_init(self.actor_lr)
        self.critic = self.critic_model_init(self.critic_lr)
        self.actor_target = keras.models.clone_model(self.actor)
        self.critic_target = keras.models.clone_model(self.critic)
        self.replay_memory = Replay_Memory()
        self.epsilon = 0.2
        self.std = 0.05

        self.actor_Adam = tf.train.AdamOptimizer(learning_rate = self.actor_lr)
        self.critic_Adam = tf.train.AdamOptimizer(learning_rate = self.critic_lr)

        self.value_grads = tf.gradients(self.critic.output, self.critic.input)
        print(self.value_grads)

        _,self.split = tf.split(self.value_grads[0],[6,2],1)
        self.param_grads = tf.gradients(
            self.actor.output,self.actor.trainable_weights,grad_ys = -self.split) 
        #self.actor_loss = -(1.0/self.batch_size) * tf.math.reduce_sum(tf.math.multiply(self.value_grads[0][:,6:],self.actor.output))
        #self.updateActor = self.actor_Adam.minimize(self.actor_loss,var_list=self.actor.trainable_weights)

        self.updateActor = self.actor_Adam.apply_gradients(zip(self.param_grads,self.actor.trainable_weights))

        self.y_values = tf.placeholder(tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.y_values,self.critic.outputs)
        self.updateCritic = self.critic_Adam.minimize(self.critic_loss,var_list = self.critic.trainable_weights)


        self.print_value_grads = tf.print(self.value_grads[0][:,6:])
        self.print_value_grads_full = tf.print(self.value_grads)
        #self.print_param_grads = tf.print(self.param_grads)
        self.print_critic_loss = tf.print(self.critic_loss)
        self.print_critic_outputs = tf.print(self.critic.output)
        self.print_actor_weights = tf.print(self.actor.trainable_weights)
        self.print_critic_weights = tf.print(self.critic.trainable_weights)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        K.set_session(self.sess)
        


    def actor_model_init(self,lr):
        model = Sequential()
        model.add(Dense(100, input_dim=6, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(2, activation='tanh'))
        #model.compile(optimizer=keras.optimizers.Adam(lr=lr),
        #        loss='MSE')
        return model

    def critic_model_init(self,lr):
        model = Sequential()
        model.add(Dense(400, input_dim=8, activation='relu'))
        model.add(Dense(400,activation='relu'))
        model.add(Dense(1, activation='linear'))
        #model.compile(optimizer=keras.optimizers.Adam(lr=lr),
        #        loss='MSE')
        return model

    def predict_action(self,state,actor_model):
        a = self.sess.run(actor_model.output,feed_dict={actor_model.input : np.array([state])})
        return a[0]

    def predict_actions(self,states,actor_model):
        a = self.sess.run(actor_model.output,feed_dict={actor_model.input : np.array(states)})
        return a

    def predict_values(self,state_actions,critic_model):
        #a = critic_model.predict(np.array(state_actions))
        a = self.sess.run(critic_model.output,feed_dict={critic_model.input : state_actions})
        return a

    def test(self, num_episodes):
        # Write function for testing here.
        # Remember you do not need to add noise to the actions
        # outputed by your actor when testing.
        rewards = []
        for _ in range(num_episodes):
            total = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.predict_action(state,self.actor)
                state,reward,done,_ = self.env.step(action)
                total += reward
            rewards.append(total)
        return np.mean(rewards)

    def add_noise(self, action, std):
        return action + np.random.normal(0,std,2)

    def train(self, num_episodes, hindsight=False):
        # Write your code here to interact with the environment.
        # For each step you take in the environment, sample a batch
        # from the experience replay and train both your actor
        # and critic networks using the sampled batch.
        #
        # If ``hindsight'' option is specified, you will use the
        # provided environment to add hallucinated transitions
        # into the experience replay buffer.
        plot_rewards = []
        for e in range(num_episodes):
            if(e % 300 == 0):
                self.save_models()
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                #TODO: make espilon independent
                #self.epsilon = .5*(1-e/60000)
                #print("state = ",state)
                if np.random.rand() < self.epsilon:
                    action = self.random_action()
                else:
                    action = self.predict_action(state,self.actor)
                    if self.verbose: print("action = ",action)
                    action = self.add_noise(action,self.std)
                action = np.clip(action,-1,1)
                newstate,reward,done,_ = self.env.step(action)
                total_reward += reward
                self.replay_memory.append((state,action,reward,newstate,done))
                transitions = self.replay_memory.sample_batch(self.batch_size)

                #get ys
                s2s = [s2 for (_,_,_,s2,_) in transitions]
                actions = np.array(self.predict_actions(s2s,self.actor_target))
                #print("actions = \n",actions)
                state_actions_2 = [concat(s2,a) for (s2,a) in zip(s2s,actions)]
                predicted_values = self.predict_values(state_actions_2,self.critic_target)
                y_values = [r+self.gamma * v if(not done) else np.array([r]) for ((s1,a,r,s2,done),v) in zip(transitions,predicted_values)]
                #print("y_vals = \n",[list(v) for v in y_values])
                state_actions = [concat(s1,a) for (s1,a,_,_,_) in transitions]
                states = [s for (s,_,_,_,_) in transitions]
                
                #pred_vals = self.predict_values(state_actions,self.critic)
                #print("predicted values = \n",pred_vals)

                #print("dif = \n",np.add(y_values,np.multiply(-1,pred_vals)))

                #print("states \n",states,"s2s = \n",s2s)

                #update the actor and critic
                size = int(self.batch_size)
                actor_inputs = {
                    self.actor.input : states[0:size],
                    self.critic.input : state_actions[0:size]}
                critic_inputs = {
                    self.critic.input : state_actions,
                    self.y_values : y_values
                }
        
                self.sess.run(self.updateActor, feed_dict=actor_inputs)
                self.sess.run(self.updateCritic,feed_dict=critic_inputs)

                #update the target weights
                self.actor_target.set_weights(
                    np.multiply(self.tau,self.actor.get_weights())
                    + np.multiply((1-self.tau),self.actor_target.get_weights()))
                self.critic_target.set_weights(
                    np.multiply(self.tau,self.critic.get_weights())
                    + np.multiply((1-self.tau),self.critic_target.get_weights()))
                state = newstate
            #print("episode = ",e,", total reward = ",total_reward)
            if(e%100 == 0):
                plot_rewards.append(self.test(30))
                plt.plot(plot_rewards,'b-')
                plt.savefig('graph_means.png',bbox_inches='tight')
        #print("------------------------")




    def random_action(self):
        return [x*2 - 1 for x in np.random.rand(2)]

    def burn_in(self, burn=10000):
        done = False
        state = self.env.reset()
        for _ in range(burn):
            if done:
                state = self.env.reset()
            action = self.random_action()
            old_state = state
            state, reward, done, _ = self.env.step(action)
            self.replay_memory.append((old_state,action,reward,state,done))

    def save_models(self):
        self.save_actor_model()
        self.save_critic_model()

    def save_actor_model(self):
        name1 = self.save_dir + self.actor_file + str(self.actor_file_count) + ".h5"
        name2 = self.save_dir + self.actor_target_file + str(self.actor_file_count) + ".h5"
        self.actor_file_count += 1
        self.actor.save_weights(name1)
        self.actor_target.save_weights(name2)

    def load_actor_model(self,actor_file,actor_target_file,file_no):
        self.actor.load_weights(self.save_dir + actor_file + str(file_no) + ".h5")
        self.actor_target.load_weights(self.save_dir + actor_target_file + str(file_no) + ".h5")

    def save_critic_model(self):
        name1 = self.save_dir + self.critic_file + str(self.critic_file_count) + ".h5"
        name2 = self.save_dir + self.critic_target_file + str(self.critic_file_count) + ".h5"
        self.critic_file_count += 1
        self.critic.save_weights(name1)
        self.critic_target.save_weights(name2)

    def load_critic_model(self,critic_file,critic_target_file,file_no):
        self.critic.load_weights(self.save_dir + critic_file + str(file_no) + ".h5")
        self.critic_target.load_weights(self.save_dir + critic_target_file + str(file_no) + ".h5")



class Replay_Memory():

    def __init__(self, memory_size=1_000_000):
        self.memory = None
        self.memsize = memory_size
        self.counter = 0
        self.full = False

    def sample_batch(self, batch_size=32):
        if self.full:
            return [self.memory[randint(0,self.memsize)] for _ in range(batch_size)]
        return [self.memory[randint(0,self.counter)] for _ in range(batch_size)]

    def append(self, transition):
        if self.memory is None:
            self.memory = [transition for _ in range(self.memsize)]
        self.memory[self.counter] = transition
        self.counter += 1
        if self.counter >= self.memsize:
            self.full = True
            self.counter = 0

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--actor_lr', dest='actor_lr', type=float,
                        default=1e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")
    parser.add_argument('--tau', dest='tau', type=float,
                        default=0.05, help="The rate to update the slow network.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.98, help="The decay of value rate.")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser.add_argument('--save_dir',dest = 'saved_dir', type = str, default = 'models/')
    parser.add_argument('--actor_file',dest='actor_file',type=str,default = 'actor_model')
    parser.add_argument('--actor_target_file',dest='actor_target_file',type=str,default = 'actor_target_model')
    parser.add_argument('--critic_file',dest='critic_file',type=str,default = 'critic_model')
    parser.add_argument('--critic_target_file',dest='critic_target_file',type=str,default = 'critic_target_model')
    parser.add_argument('--test',dest='test',type=int,default = 0)
    parser.add_argument('--step',dest='step',type=int,default = 5)
    parser.add_argument('--train_from',dest='train_from',type=int,default = 0)
    parser.add_argument('--graph',dest='graph',type=int,default = 0)
    parser.add_argument('--verbose', dest='verbose',
                              action='store_true')
    return parser.parse_args()

def main():
    #tf.enable_eager_execution()

    args = parse_arguments()
    env = gym.make('Pushing2D-v0')
    algo = DDPG(env, args)
    algo.train(50000) #50000
    print(algo.test(100))
    '''rewards = []
    for e in range(100):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = [.5,.5]
            newstate,reward,done,_ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    print(np.mean(rewards))'''
        

if __name__=='__main__':
    main()
