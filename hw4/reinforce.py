import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        model.compile(optimizer=keras.optimizers.Adam(lr=lr),
              loss=self.customLoss)
        #self.AdamOpt = tf.train.AdamOptimizer(learning_rate = lr)
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

    def train(self, env, gamma=1.0,render=False):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states,actions,rewards = self.generate_episode(env,render=render)
        rewards = np.multiply(rewards,1/100)
        last_reward = 0
        T = len(states)
        G_t = np.zeros(T).tolist()
        #predicted = np.zeros(T).tolist()
        for t in reversed(list(range(T))):
            reward = rewards[t] + last_reward*gamma
            last_reward = reward
            #predicted[t] = tf.math.scalar_mul(reward/T, K.log(actions[t]))
            G_t[t] = reward/T
        #print(actions)
        self.model.fit(x = np.array(states), y=np.array(G_t),verbose=0)

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        state = env.reset()
        done = False
        while not done:
            if(render):
                env.render()
            states.append(state)
            action = self.predict_action(state)
            actions.append(action)
            state,reward,done,_ = env.step(action)
            rewards.append(reward)
        rewards = np.array(rewards,dtype='float')
        print(np.sum(rewards))
        if(np.sum(rewards) > 150): exit(0)
        return states, actions, rewards

    @staticmethod
    def customLoss(yTrue,yPred):
        #L = 0
        G_t = yTrue
        print(G_t, yPred)
        #return K.sum(K.prod(G_t,K.log(yPred)),keepdims=True)
        return K.sum(yPred)
        '''for i in range(len(yTrue)):
            L += tf.math.scalar_mul(yTrue[i], K.log(yPred[i]))
        return L'''

    def predict_action(self,state):
        s = [state]
        a = self.model.predict(np.array(s))
        return np.random.choice(len(a),a)

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


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    r = Reinforce(model,lr)
    for i in range(10**5):
        print("iteration = ",i)
        r.train(env,render=render)


if __name__ == '__main__':
    main(sys.argv)
