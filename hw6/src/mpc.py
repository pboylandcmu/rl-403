import os
import tensorflow as tf
import numpy as np
import math
import time

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
        self.D = []
        self.reset()

        #training the state predictor
        #PSEUDOCODE
        '''nextmeans, nextvars = model.output[:, 0:self.state_dim], output[:, self.state_dim:]
        realnext = tf.placeholder(tf.float32,shape=(12,128))
        mydists = tf.distributions.Normal(loc = nextmeans,scale=nextvars)
        likelihoods = mydists.pdf(realnext)
        self.loss = -1 * tf.math.log(tf.reduce_prod(likelihoods))'''

    def obs_cost_fn(self, state,goal):
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5
        W_DIR = -5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = goal[0], goal[1] 
        #box_vel = state[6],state[7]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        #dir = box_x-pusher_x,box_y-pusher_y

        #dot = np.dot(dir,box_vel)/(np.linalg.norm(dir)*np.linalg.norm(box_vel))

        if((pusher_x > 4.5 or pusher_x < .5 or pusher_y > 4.5 or pusher_y < .5)):
          out_of_bounds = 10000
        elif ((box_x > 4.5 or box_x < 0.5 or box_y > 4.5 or box_y < .5)):
          out_of_bounds = 1000
        else: out_of_bounds = 0
        if(goal_x > 4.5 or goal_x < 0.5 or goal_y > 4.5 or goal_y < .5):
          out_of_bounds = 0
        
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box-0.4,0) + W_GOAL * d_goal + W_DIFF * diff_coord + out_of_bounds


    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (note this may not be used)
          epochs: number of epochs to train for
        """
        for i in range(len(obs_trajs)):
          for j in range(len(obs_trajs[i]) - 1):
            s,a,ns = obs_trajs[i][j],acs_trajs[i][j],obs_trajs[i][j+1]
            s = s.copy()
            s = s[0:8]
            ns = ns.copy()
            ns = ns[0:8]
            self.D.append((s,a,ns))
        self.model.train(self.D,epochs)

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
        self.CEM(200,20,5,state)
        a = self.mu[0,:]
        #print("--------------------------")
        #print("state: ",state)
        #print("t: ",t," a: ",a)
        self.mu = list(self.mu[1:])
        self.mu.append([0,0])
        self.mu = np.array(self.mu)
        return a

    def CEM(self,pop_size,num_elites,iters,state):
      #destructively modifies mu and sigma
      #print("starting CEM")
      #start = time.time()
      goal = [state[8],state[9]]
      state = state.copy()
      state = state[0:8]
      #TS = self.TS1()
      for _ in range(iters):
        #print("-------------------")
        action_sequences = [np.random.normal(self.mu,self.sigma) for _ in range(pop_size)]
        costs = np.zeros(pop_size)

        ss = [(state,anum,p,0) for p in range(self.npart) for anum in range(pop_size)]
        ts = [self.TS1() for _ in range(pop_size)]
        while not all(t >= self.plan_hor - 1 for (_,_,_,t) in ss):
          oss = [(s,a,p,t,i) for ((s,a,p,t),i) in zip(ss,range(len(ss))) if t<(self.plan_hor-1)]
          models = [ts[anum][p][t] for (_,anum,p,t,_) in oss]
          #only works when only 2 models
          mtorun = 0 if (np.mean(models)<0.5) else 1
          sitorun = [(s,i) for ((s,_,_,t,i),m) in zip(oss,models) if m == mtorun]
          #print(len(sitorun))
          #print(ss)
          storun = [s for (s,i) in sitorun]
          sstorun = [ss[i] for (s,i) in sitorun]
          atorun = [action_sequences[a][t] for (_,a,_,t) in sstorun]
          mean,std = self.model.predict(mtorun,storun,atorun)
          newst = [(s+d,i) for ((s,i),d) in zip(sitorun,np.random.normal(mean,std))]
          for (news,i) in newst:
            (s,a,p,t) = ss[i]
            ss[i] = (news,a,p,t+1)
            costs[a] += self.obs_cost_fn(news,goal)
        '''
        for model_row in range(self.npart):
          s = [state for _ in range(pop_size)]
          for model_col in range(self.plan_hor):
            model_num = TS[model_row][model_col]
            a = [action_sequence[model_col] for action_sequence in action_sequences]
            mean,std = self.model.predict(model_num,s,a)
            #std = self.model.sess.run(
            #  tf.math.sqrt(tf.math.exp(logvar)))
            s = s + np.random.normal(mean,std)
            for c in range(pop_size):
              costs[c] += self.obs_cost_fn(s[c],goal)
        '''
        #print(costs[0:10])
        elites = self.get_elites(costs,num_elites)
        best_mus = [action_sequences[e] for e in elites]
        self.mu = np.mean(best_mus,axis = 0)
        self.sigma = np.std(best_mus,axis = 0)
      #end = time.time()
      #print("ending CEM: ",end-start)

    def get_elites(self,costs,num_elites):
      #returns the indeces of the top e lowest costs
      elites = []
      costs_copy = costs.copy()
      costs_copy.sort()
      elites_costs = costs_copy[0:num_elites]
      for c in elites_costs:
        elites.append(list(costs).index(c))
      return elites
        
         

    def TS1(self):
      N = self.num_nets
      P = self.npart
      T = self.plan_hor
      return np.random.randint(0,high = N,size = (P,T))














