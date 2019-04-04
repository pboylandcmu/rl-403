import gym
import envs
from ddpg import DDPG

def main():
    env = gym.make('Pushing2D-v0')
    kwargs = {}
    algo = DDPG(env, **kwargs)
    algo.train(50000)

if __name__=='__main__':
    main()
