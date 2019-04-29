import os
import tensorflow as tf
import numpy as np

class MPC:
    def __init__(self, env, npart, plan_hor, model, popsize, num_elites, max_iters):
        """
        Arguments:
          env: environment to interact with
          npart: number of particles
          plan_hor: plan horizon
        """
        self.env = env
        self.npart = npart
        self.plan_hor = plan_hor
        self.num_nets = model.num_nets
        self.model = model

        # Set up optimizer
        self.optimizer = #Initialize your planner with the relevant arguments.


    def obs_cost_fn(self, state):
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box-0.4,0) + W_GOAL * d_goal + W_DIFF * diff_coord


    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (note this may not be used)
          epochs: number of epochs to train for
        """


    def reset(self):
        """
        Perform any cleanup for MPC that you need before running for a new episode.
        """


    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        Return:
          action from MPC
        """










