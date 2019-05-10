import numpy as np
import gym
import envs

from agent import Agent
from agent import RandomPolicy
from mpc import MPC
from model import PENN

# Training params
TASK_HORIZON = 40
#NUM_PARTICLES = 6
NUM_PARTICLES = 1
PLAN_HOR = 5
#NUM_NETS = 2
NUM_NETS = 1

NTRAIN_ITERS = 501
NROLLOUTS_PER_ITER = 1
#NINIT_ROLLOUTS = 100
NINIT_ROLLOUTS = 1000

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8
ACTION_DIM = 2

class Experiment:
    def __init__(self):
        self.env = gym.make('Pushing2D-v1')
        self.task_hor = TASK_HORIZON

        self.agent = Agent(self.env)
        self.model = PENN(NUM_NETS, STATE_DIM, ACTION_DIM, LR)
        self.policy = MPC(self.env, NUM_PARTICLES, PLAN_HOR, self.model, POPSIZE, NUM_ELITES, MAX_ITERS)


    def test(self, num_episodes):
        samples = []
        for j in range(num_episodes):
            samples.append(
                self.agent.sample(
                    self.task_hor, self.policy
                )
            )
        print("Rewards obtained:", np.mean([sample["reward_sum"] for sample in samples]))
        print("Percent success:", np.mean([sample["rewards"][-1]==0 for sample in samples]))
        return np.mean([sample["rewards"][-1]==0 for sample in samples])

    def train(self):
        traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []
        test_results = []
        samples = []
        rand_pol = RandomPolicy(2)
        for i in range(NINIT_ROLLOUTS):
            samples.append(self.agent.sample(self.task_hor, rand_pol))
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])

        if NINIT_ROLLOUTS>0:
            self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples],
                    epochs=10
            )

        for i in range(NTRAIN_ITERS):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            samples = []
            for j in range(NROLLOUTS_PER_ITER):
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
            print("Rewards obtained:", [sample["reward_sum"] for sample in samples])
            traj_obs.extend([sample["obs"] for sample in samples])
            traj_acs.extend([sample["ac"] for sample in samples])
            traj_rets.extend([sample["reward_sum"] for sample in samples])
            traj_rews.extend([sample["rewards"] for sample in samples])

            if(i % 50 == 0):
                self.model.save_models()
                test_results.append((i,self.test(20)))
                test_file = open("test_graph.txt","w")
                test_file.writelines([str(epoch) + "," + str(result) + "\n" for (epoch,result) in test_results])
                test_file.close()

            self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
            )
            

if __name__=="__main__":
    print("task horizon: %d, # particles: %d, plan horizon: %d, num_nets: %d, # init rollouts: %d" % (TASK_HORIZON, NUM_PARTICLES, PLAN_HOR, NUM_NETS, NINIT_ROLLOUTS))
    print("popsize: %d, # elites: %d, max iters: %d" % (POPSIZE, NUM_ELITES, MAX_ITERS))
    exp = Experiment()
    #exp.train()
    exp.model.load_models(9)
    exp.test(20)

