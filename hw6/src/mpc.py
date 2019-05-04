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
        self.reset()

        #training the state predictor
        #PSEUDOCODE
        '''nextmeans, nextvars = model.output[:, 0:self.state_dim], output[:, self.state_dim:]
        realnext = tf.placeholder(tf.float32,shape=(12,128))
        mydists = tf.distributions.Normal(loc = nextmeans,scale=nextvars)
        likelihoods = mydists.pdf(realnext)
        self.loss = -1 * tf.math.log(tf.reduce_prod(likelihoods))'''


    def obs_cost_fn(self, state):
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = state[4], state[5] #needs to be checked

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
        pass


    def reset(self):
        """
        Perform any cleanup for MPC that you need before running for a new episode.
        """
        self.mu = np.zeros((self.plan_hor,2))
        self.sigma = np.ones((self.plan_hor,2))


    def act(self, state, t):
        """
        Use model predictive control to find the action given current state.

        Arguments:
          state: current state
          t: current timestep
        Return:
          action from MPC
        """
        self.CEM(200,20,5,self.mu,self.sigma,state)
        a = self.mu[0,:]
        self.mu = self.mu[1:].append([0,0])
        return a

    def CEM(self,pop_size,num_elites,iters,mu,sigma,state):
      #destructively modifies mu and sigma
      TS = self.TS1()
      for _ in range(iters):
        action_sequences = [np.random.normal(mu,sigma) for _ in range(pop_size)]
        costs = np.zeros(pop_size)
        for _ in range(pop_size):
          s = [state for _ in range(pop_size)]
          for model_row in range(self.npart):
            for model_col in range(self.plan_hor):
              model_num = TS[model_row][model_col]
              a = [action_sequence[model_col] for action_sequence in action_sequences]
              s = self.model.predict(model_num,s,a)
              for c in range(pop_size):
                costs[c] += self.obs_cost_fn(s[c])
        elites = self.get_elites(costs,num_elites)
        best_mus = [action_sequences[e] for e in elites]
        mu = np.mean(best_mus,axis = 0)
        sigma = np.std(best_mus,axis = 0)

    def get_elites(self,costs,num_elites):
      #returns the indeces of the top e lowest costs
      elites = []
      for _ in range(num_elites):
        elites.append(float('inf'))
      for c in costs:
        for e in range(num_elites):
          if(c < elites[e]):
            elites[e] = c
            break
      for e in range(num_elites):
        elites[e] = costs.index(elites[e])
      return elites
        
         

    def TS1(self):
      N = self.num_nets
      P = self.npart
      T = self.plan_hor
      return np.random.randint(0,high = N,size = (P,T))














