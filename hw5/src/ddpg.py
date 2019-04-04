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
        self.train_from = args.train_from
        self.verbose = args.verbose
        self.env = env

        self.actor = self.actor_model_init(self.actor_lr)
        self.critic = self.critic_model_init(self.critic_lr)
        self.actor_target = keras.models.clone_model(self.actor)
        self.critic_target = keras.models.clone_model(self.critic)


    def actor_model_init(self,lr):
        model = Sequential()
        model.add(Dense(16, input_dim=6, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='tanh'))
        model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                loss='MSE')
        return model

    def critic_model_init(self,lr):
        model = Sequential()
        model.add(Dense(16, input_dim=8, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                loss='MSE')
        return model

    def generate_episode(self,process):
        states = []
        actions = []
        rewards = []
        values = []
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
        if self.verbose: print("total reward = ",np.sum(rewards),", ep length = ",len(rewards))
        return states, actions, rewards

    def predict_action(self,state):
        a = self.actor.predict(np.array([state]))
        return a[0] # + some noise???

    def test(self, num_episodes):
        # Write function for testing here.
        # Remember you do not need to add noise to the actions
        # outputed by your actor when testing.


    def train(self, num_episodes, hindsight=False):
        # Write your code here to interact with the environment.
        # For each step you take in the environment, sample a batch
        # from the experience replay and train both your actor
        # and critic networks using the sampled batch.
        #
        # If ``hindsight'' option is specified, you will use the
        # provided environment to add hallucinated transitions
        # into the experience replay buffer.



    def add_hindsight_replay_experience(self, states, actions, end_state):
        # Create transitions for hindsight experience replay and
        # store into replay memory.
        # into the experience replay buffer.    

    def burn_in(self, burn=10000):
        done = False
        state = self.env.reset()
        for _ in range(burn):
            if done:
                state = self.env.reset()
            action = np.random.rand(2)
            old_state = state
            state, reward, done, _ = self.env.step(action)
            self.replay_memory.append((old_state,action,reward,state,done))


class Replay_Memory():

    def __init__(self, memory_size=50000):
        self.memory = None
        self.memsize = memory_size
        self.counter = 0
        self.full = False

        pass

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
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser.add_argument('--actor_file',dest='actor_file',type=str,default = 'actor_models/model')
    parser.add_argument('--actor_target_file',dest='actor_target_file',type=str,default = 'actor_models/target_model')
    parser.add_argument('--critic_file',dest='critic_file',type=str,default = 'critic_models/model')
    parser.add_argument('--critic_target_file',dest='critic_target_file',type=str,default = 'critic_models/target_model')
    parser.add_argument('--test',dest='test',type=int,default = 0)
    parser.add_argument('--step',dest='step',type=int,default = 5)
    parser.add_argument('--train_from',dest='train_from',type=int,default = 0)
    parser.add_argument('--graph',dest='graph',type=int,default = 0)
    parser.add_argument('--verbose', dest='verbose',
                              action='store_true')
    return parser.parse_args()

def main():
    args = parse_arguments()
    env = gym.make('Pushing2D-v0')
    algo = DDPG(env, args)
    algo.train(50000)

if __name__=='__main__':
    main()
