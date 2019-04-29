import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """
    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        Arguments:
          sess: tensorflow session
          num_nets: number of networks in the ensemble
          state_dim: state dimension
          action_dim: action dimension
        """

        self.sess = tf.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        K.set_session(self.sess)
        self.lr = learning_rate

        # Log variance bounds
        self.max_logvar = tf.Variable(-3*np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7*np.ones([1, self.state_dim]), dtype=tf.float32)
        self.models = [self.create_network() for _ in range(self.num_nets)]


    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2*self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(input=I,output=O)
        return model

