import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import load_model
import gym
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time



class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, model_file, env, train_from):
        self.model = model
        self.model.compile(optimizer=keras.optimizers.Adam(lr=lr),loss=self.customLoss)
        self.action_size = 4
        self.file_count = train_from
        self.model_file = model_file
        self.env = env
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

    def train(self, gamma=1.0,render=False,baseline=0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states,actions,rewards = self.generate_episode(render=render)
        T = len(states)
        rewards = np.multiply(rewards,1.0/(100 * T))
        last_reward = 0
        G_t = [0] * T
        yTrue = [0] * T
        for t in reversed(list(range(T))):
            reward = rewards[t] + last_reward * gamma
            last_reward = reward
            G_t[t] = reward
            v = np.zeros(self.action_size)
            v[actions[t]] = reward #- baseline / T
            yTrue[t] = v

        self.model.train_on_batch(x = np.array(states), y=np.array(yTrue))
        return np.mean(G_t) * T

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

    @staticmethod
    def customLoss(yTrue,yPred):
        return -K.sum(K.sum(tf.multiply(yTrue,K.log(yPred))))

    def predict_action(self,state):
        s = [state]
        a = self.model.predict(np.array(s))
        #return np.argmax(a[0])
        return np.random.choice(range(self.action_size),p=a[0])

    def predict_action_verbose(self,states):
        a = self.model.predict(np.array(states))
        print(a)
        #for r in a:
        #    print(r)

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
            if(verbosity == 1 and len(reward) == 1000):
                self.predict_action_verbose(states)
            if(e == 0 and verbosity == 1):
                self.predict_action_verbose(states)
        return np.mean(rewards), np.std(rewards)

    def graph(self,graph,step=5):
        means = []
        stds = []
        for i in range(0,graph,step):
            self.load_model(self.model_file,i)
            mean, std = self.test(episodes=100,render=False,verbosity=0)
            means.append(mean)
            stds.append(std)
        plt.plot(range(0,graph,step),means)
        plt.xlabel("model number")
        plt.ylabel("average reward")
        plt.title("Lunar Lander REINFORCE Performance plot")
        plt.show()





def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.add_argument('--verbose', dest='verbose',
                              action='store_true')
    parser.set_defaults(render=False,verbose=False)

    parser.add_argument('--model_file',dest='model_file',type=str,default = 'models/model')
    parser.add_argument('--test',dest='test',type=int,default = 0)
    parser.add_argument('--graph',dest='graph',type=int,default = 0)
    parser.add_argument('--step',dest='step',type=int,default = 5)
    parser.add_argument('--train_from',dest='train_from',type=int,default = 0)
    parser.add_argument('--graph',dest='graph',type=int,default = 0)

    return parser.parse_args()

def runningAverage(rewards):
    if len(rewards) < 100:
        return np.mean(rewards)
    return np.mean(rewards[-100:])

def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render
    model_file = args.model_file
    test = args.test
    verbose = args.verbose
    train_from = args.train_from
    graph = args.graph

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    r = Reinforce(model,lr,model_file,env,train_from)
    if(graph):
        r.graph(graph,step=args.step)
    elif(not test):
        if(train_from):
            r.load_model(model_file,train_from)
        rewards = []
        baseline = 0
        for i in range(train_from*100,num_episodes+1):
            print("iteration = ",i, ", baseline = ", baseline)
            rewards.append(r.train(render=render,baseline = baseline))
            if(i % 100 == 0):
                r.save_model()
                baseline = runningAverage(rewards)
    else:
        r.load_model(model_file,test)
        print(r.test(verbosity = verbose,render=render))



if __name__ == '__main__':
    main(sys.argv)
