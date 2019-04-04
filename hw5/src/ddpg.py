import numpy as np
import argparse



class DDPG:
    def __init__(self, env, **kwargs):
        # Initialize your class here with relevant arguments
        # e.g. learning rate for actor and critic, update speed
        # for target weights, etc.


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

