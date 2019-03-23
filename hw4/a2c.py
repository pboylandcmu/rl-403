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


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
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

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        return

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

    @staticmethod
    def customLoss(yTrue,yPred):
        return -K.sum(K.sum(tf.multiply(yTrue,K.log(yPred))))

    def save_model(self,model_file=None):
        if(model_file is None):
            name = self.model_file + str(self.file_count) + ".h5"
        else:
            name = model_file + str(self.file_count) + ".h5"
        self.file_count += 1
        self.model.save_weights(name)
        return name

    def load_model(self,model_file,file_no):
        self.model.load_weights(model_file + str(file_no) + ".h5")

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

    return parser.parse_args()

def make_critic():
    model = Sequential()
    model.add(Dense(16, input_dim=4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
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

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
