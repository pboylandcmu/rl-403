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

        #creating the models
        self.actorInput,self.actorOutput,self.actorWeights = self.actor_model_init()
        self.criticInput,self.criticOutput,self.criticWeights = self.critic_model_init(self.actorOutput,self.actorInput)
        self.actorTargetInput,self.actorTargetOutput,self.actorTargetWeights = self.actorInput,self.actorOutput,self.actorWeights #self.actor_model_init()
        self.criticTargetInput,self.criticTargetOutput,self.criticTargetWeights = self.criticInput,self.criticOutput,self.criticWeights #self.critic_model_init(self.actorTargetOutput,self.actorTargetInput)

        #creating the losses and optimizers
        self.AdamCritic = tf.train.AdamOptimizer(learning_rate = self.critic_lr)
        self.AdamActor = tf.train.AdamOptimizer(learning_rate = self.actor_lr)

        self.y_value = tf.placeholder(tf.float32,shape=None)
        criticLoss = tf.losses.mean_squared_error(self.y_value,self.criticOutput)
        self.trainCritic = self.AdamCritic.minimize(criticLoss)

        actorGradients = tf.gradients(self.criticOutput,self.actorWeights)
        self.trainActor = self.AdamActor.apply_gradients(list(zip(actorGradients,self.actorWeights)))

        #creating the sets
        #self.copyActor = tf.assign(np.array(self.actorTargetWeights),np.array(self.actorWeights))
        #self.copyCritic = tf.assign(self.criticTargetWeights,np.array(self.criticWeights))

        #preparing to run and train
        self.replay_memory = Replay_Memory()
        self.epsilon = 0.2
        self.std = 0.01

        '''
        

        self.value_grads = tf.gradients(self.critic.output, self.critic.input)
        #print(self.value_grads[0][0][6:])
        self.param_grads = tf.gradients(self.actor.output,self.actor.trainable_weights,grad_ys = self.value_grads[0][0][6:])

        self.updateActor = self.Adam.apply_gradients(list(zip(self.param_grads,self.actor.trainable_weights)))

        self.myprint = tf.print(self.param_grads)'''

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def actor_model_init(self):
        with tf.variable_scope("actorscope"):
            actorInput = tf.placeholder(tf.float32, shape=(None,6))
            layer1 = Dense(400,activation='relu')(actorInput)
            layer2 = Dense(16,activation='relu')(layer1)
            actorOutput = Dense(2,activation='tanh')(layer2)
            #actorOutput = [var for var in actorOutput.output]
            actorWeights = tf.trainable_variables(scope="actorscope")
            return actorInput, actorOutput, actorWeights

    def critic_model_init(self,actorInput,actorOutput):
        with tf.variable_scope("criticscope"):
            criticInput = tf.concat([actorInput,actorOutput],-1)
            layer1 = Dense(400,activation='relu')(criticInput)
            layer2 = Dense(16,activation='relu')(layer1)
            criticOutput = Dense(2,activation='linear')(layer2)
            #criticOutput = [var for var in criticOutput.output]
            criticWeights = tf.trainable_variables(scope="criticscope")
            return criticInput, criticOutput, criticWeights

    def choose(self,state):
        a = self.sess.run(self.actorOutput,feed_dict={self.actorInput: np.array([state])})
        return a[0]

    def predict_value(self,state):
        a = self.sess.run(self.criticTargetOutput,feed_dict={self.actorTargetInput: np.array([state])})
        return a[0][0]

    def test(self, num_episodes):
        rewards = []
        for _ in range(num_episodes):
            total = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose(state)
                state,reward,done,_ = self.env.step(action)
                total += reward
            rewards.append(total)
        return np.mean(rewards)

    def add_noise(self, action):
        return action + np.random.normal(0,.01,2)

    def calc_y_value(self,reward,state):
        return reward + self.gamma * self.predict_value(state)

    def train(self, num_episodes, hindsight=False):
        # Write your code here to interact with the environment.
        # For each step you take in the environment, sample a batch
        # from the experience replay and train both your actor
        # and critic networks using the sampled batch.
        #
        # If ``hindsight'' option is specified, you will use the
        # provided environment to add hallucinated transitions
        # into the experience replay buffer.
        for e in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if np.random.rand() < self.epsilon:
                    action = self.random_action()
                else:
                    action = self.choose(state)
                    action = self.add_noise(action)
                newstate,reward,done,_ = self.env.step(action)
                total_reward += reward
                self.replay_memory.append((state,action,reward,newstate))

                for _ in range(32):
                #setting up lists
                    transitions = self.replay_memory.sample_batch(1)
                    y_values = [self.calc_y_value(r,s) for (_,_,r,s) in transitions]
                    state_actions = [concat(s1,a) for (s1,a,_,_) in transitions]
                    states = [s for (s,_,_,_) in transitions]

                    #update the critic and actor
                    '''criticInput:state_actions'''
                    #TODO: not sure if this is running correctly
                    self.sess.run(self.trainCritic,feed_dict={self.criticInput:state_actions,self.actorInput:states, self.y_value:y_values})
                    self.sess.run(self.trainActor,feed_dict={self.actorInput:states})

            #self.sess.run(self.copyCritic)
            #self.sess.run(self.copyActor)

            print("episode = ",e,", total reward = ",total_reward)



    def random_action(self):
        return [x*2 - 1 for x in np.random.rand(2)]

    def burn_in(self, burn=100):
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
        name1 = self.actor_file + str(self.save_dir + self.actor_file_count) + ".h5"
        name2 = self.actor_target_file + str(self.save_dir + self.actor_file_count) + ".h5"
        self.actor_file_count += 1
        self.actor.save_weights(name1)
        self.actor_target.save_weights(name2)

    def load_actor_model(self,actor_file,actor_target_file,file_no):
        self.actor.load_weights(self.save_dir + actor_file + str(file_no) + ".h5")
        self.actor_target.load_weights(self.save_dir + actor_target_file + str(file_no) + ".h5")

    def save_critic_model(self):
        name1 = self.critic_file + str(self.save_dir + self.critic_file_count) + ".h5"
        name2 = self.critic_target_file + str(self.save_dir + self.critic_file_count) + ".h5"
        self.critic_file_count += 1
        self.critic.save_weights(name1)
        self.critic_target.save_weights(name2)

    def load_critic_model(self,critic_file,critic_target_file,file_no):
        self.critic.load_weights(self.save_dir + critic_file + str(file_no) + ".h5")
        self.critic_target.load_weights(self.save_dir + critic_target_file + str(file_no) + ".h5")



class Replay_Memory():

    def __init__(self, memory_size=5000):
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
                        default=1.0, help="The decay of value rate.")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser.add_argument('--save_dir',dest = 'saved_dir', type = str, default = 'models')
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

if __name__=='__main__':
    main()
