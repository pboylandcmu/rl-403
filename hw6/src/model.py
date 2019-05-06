import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from numpy.random import randint

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
        self.outputs = [self.get_output(model.output) for model in self.models]
        self.means = [mean for (mean,_) in self.outputs]
        self.logvars = [logvar for (_,logvar) in self.outputs]
        self.optimizers = [tf.train.AdamOptimizer(learning_rate = learning_rate) for _ in range(self.num_nets)]
        self.state_in = tf.placeholder(tf.float32)
        self.state_out = tf.placeholder(tf.float32)
        self.losses = [tf.reduce_sum(
          tf.linalg.matmul(tf.math.reciprocal(tf.math.exp(self.logvars[i])),
          tf.math.square(
            tf.math.add(
              self.state_in,tf.math.subtract(self.means[i],self.state_out))),
          transpose_b= True)
          + tf.math.log(
            tf.math.reduce_prod(tf.math.exp(self.logvars[i]))),axis = 1)
          for i in range(self.num_nets)]
        
        self.updates = [op.minimize(loss,var_list = model.trainable_weights) 
          for (op,loss,model) in zip(self.optimizers,self.losses,self.models)]
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        '''
        a simple test for shape correctness
        f = {self.models[0].input : [[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]],
            self.state_placeholder : [[1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.]]
            }
        self.sess.run(tf.print(self.losses[0]),f)
        exit(0)'''
        
    def update_net(self,index,state_in,state_action_in,state_out):
      feed = {self.state_in : state_in,
            self.state_out : state_out,
            self.models[index].inputs : state_action_in}
      self.sess.run(self.updates[index],feed)

    def concat(self,list1,list2):
      result = []
      for e in list1:
          result.append(e)
      for e in list2:
          result.append(e)
      return result

    def train(self,D,epochs):
      #D = (state,action,nextState)
      batch_size = 128
      size = len(D)
      train_on = [[D[randint(0,size)] for _ in range(size)] for _ in range(self.num_nets)]
      for _ in range(epochs):
        for n in range(self.num_nets):
          data = train_on[n]
          np.random.shuffle(data)
          for i in range(0,len(data),batch_size):
            train = data[i:np.min(i+batch_size,len(data))]
            state = train[:,0]
            action = train[:,1]
            nextState = train[:,2]
            state_action = [self.concat(s,a) for (s,a) in zip(state,action)]
            self.update_net(n,state,state_action,nextState)

    def predict(self,index,states,actions):
      state_actions = np.array([self.concat(state,action) for (state,action) in zip(states,actions)])
      model = self.models[index]
      feed = {model.input:state_actions}
      return self.get_output(self.sess.run(model.output,feed))


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

