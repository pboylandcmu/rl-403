import gym
import envs
from ddpg import DDPG

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
