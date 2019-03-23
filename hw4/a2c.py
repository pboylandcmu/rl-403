import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import gym

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


class A2C(object):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr,actor_file,critic_file,env, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n
        self.actor_file_count = 0
        self.critic_file_count = 0
        self.actor_model_file = actor_file
        self.critic_model_file = critic_file
        self.env = env
        self.action_size = 4

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

    @staticmethod
    def r2R(rewards, n):
        kern = np.ones(n)
        convd = np.convolve(rewards[::-1],kern,'full')[:len(rewards)]
        return convd

    def getReward(self,states,t):
        if (t > len(states)):
            return 0
        else:
            return self.predict_value(states[t])

    def train(self, gamma=1.0,render=False):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        n = self.n+1
        states,actions,rewards = self.generate_episode(render=render)
        T = len(states)
        rewards = np.multiply(rewards,1.0/(100))
        yTrue = [0] * T
        convRewards = list(reversed(A2C.r2R(rewards,n)))
        for t in reversed(list(range(T))):
            v = np.zeros(self.action_size)
            v[actions[t]] = convRewards[t] + self.getReward(states,t+n) - self.getReward(states,t)
            yTrue[t] = v
        self.model.train_on_batch(x = np.array(states), y=np.array(yTrue))
        self.critic_model.train_on_batch(x = np.array(states), y=convRewards)

    def generate_episode(self, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        done = False
        while not done:
            if(render):
                self.env.render()
            states.append(state)
            action = self.predict_action(state)
            actions.append(action)
            state,reward,done,_ = self.env.step(action)
            rewards.append(reward)
        rewards = np.array(rewards,dtype='float')
        #print("total reward = ",np.sum(rewards),", ep length = ",len(rewards))
        #print(rewards)
        return states, actions, rewards

    def predict_action(self,state):
        s = [state]
        a = self.model.predict(np.array(s))
        #return np.argmax(a[0])
        return np.random.choice(range(self.action_size),p=a[0])

    def predict_value(self,state):
        s = [state]
        a = self.critic_model.predict(np.array(s))
        #return np.argmax(a[0])
        return a[0][0]

    @staticmethod
    def customLoss(yTrue,yPred):
        return -K.sum(K.sum(tf.multiply(yTrue,K.log(yPred))))

    def save_models(self):
        self.save_actor_model()
        self.save_critic_model()

    def save_actor_model(self,model_file=None):
        if(model_file is None):
            name = self.actor_model_file + str(self.actor_file_count) + ".h5"
        else:
            name = model_file + str(self.actor_file_count) + ".h5"
        self.actor_file_count += 1
        self.model.save_weights(name)
        return name

    def load_actor_model(self,model_file,file_no):
        self.model.load_weights(model_file + str(file_no) + ".h5")

    def save_critic_model(self,model_file=None):
        if(model_file is None):
            name = self.critic_model_file + str(self.critic_file_count) + ".h5"
        else:
            name = model_file + str(self.critic_file_count) + ".h5"
        self.critic_file_count += 1
        self.critic_model.save_weights(name)
        return name

    def load_critic_model(self,model_file,file_no):
        self.critic_model.load_weights(model_file + str(file_no) + ".h5")

    def test(self,episodes=100,verbosity = 0,render = False):
        rewards = []
        for e in range(episodes):
            states, actions, reward = self.generate_episode(render=render)
            rewards.append(np.sum(reward))
        return np.mean(rewards), np.std(rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser.add_argument('--actor_file',dest='actor_file',type=str,default = 'actor_models/model')
    parser.add_argument('--critic_file',dest='critic_file',type=str,default = 'critic_models/model')
    parser.add_argument('--test',dest='test',type=int,default = 0)
    parser.add_argument('--step',dest='step',type=int,default = 5)
    parser.add_argument('--train_from',dest='train_from',type=int,default = 0)
    parser.add_argument('--graph',dest='graph',type=int,default = 0)
    parser.add_argument('--verbose', dest='verbose',
                              action='store_true')
    return parser.parse_args()

def make_critic(lr):
    model = Sequential()
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
            loss='MSE')
    return model

def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render
    actor_file = args.actor_file
    critic_file = args.critic_file
    test = args.test
    verbose = args.verbose
    train_from = args.train_from
    graph = args.graph

    # Create the environment.
    env = gym.make('LunarLander-v2')
    critic = make_critic(critic_lr)
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using A2C and plot the learning curves.
    a2c = A2C(model,lr,critic,critic_lr,actor_file,critic_file,env,n=n)
    if(not test and not graph):
        if(train_from):
            a2c.load_actor_model(actor_file,train_from)
            a2c.load_critic_model(critic_file,train_from)
        for i in range(train_from*100,num_episodes+1):
            print("iteration = ",i)
            a2c.train(render=render)
            if(i % 100 == 0):
                a2c.save_models()
    elif graph():
        pass
    elif(test):
        a2c.load_actor_model(actor_file,test)
        a2c.load_critic_model(critic_file,test)
        print(a2c.test(verbosity = verbose,render = render))


if __name__ == '__main__':
    main(sys.argv)
