from env import sudoku

## Importing personal environment for playing Sudoku

env = sudoku()

import tensorflow as tf
import os
import numpy as np
import numpy.random as rnd

## Importing Libraries

def q_network(X_state,name):## Defining a neaural network to find the q value for an observation

    ## args:
        ## X_state : Input the state shape of (9,9,9)
        ## name    : Name of the network

    ## output:
        ## outputs : output layer values shape of(9**3)
        ## trainable_vars_by_name : name of the each neauron in each layers
  with tf.variable_scope(name) as scope:
    conv1 = tf.layers.conv2d(X_state,filters=9,kernel_size=(9,1),activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1,filters=9,kernel_size=(1,9),activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2,filters=9,kernel_size=(1,1),activation=tf.nn.relu)
    mid = tf.reshape(conv3,shape=[-1,9])
    fc1 = tf.layers.dense(mid,9**2)
    fc1_norm = tf.layers.batch_normalization(fc1,training=False,momentum=0.9)
    fc1_act = tf.nn.relu(fc1_norm)
    fc2 = tf.layers.dense(fc1_act,9**3)
    fc2_norm = tf.layers.batch_normalization(fc2,training=False,momentum=0.9)
    fc2_act = tf.nn.relu(fc2_norm)
    outputs = tf.layers.dense(fc2_act,9**3)
    
  trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope.name)
  trainable_vars_by_name =  {var.name[len(scope.name):]: var
                            for var in trainable_vars}
  return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None,9,9,9])
actor_q_values,actor_vars = q_network(X_state,name="q_networks/actor")
## Actor q value to make decission
critic_q_values,critic_vars = q_network(X_state,name="q_networks/critic")
## Critic q value to guess actor q value and increse rewards
copy_ops = [actor_var.assign(critic_vars[var_name])
           for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32,shape=[None])
q_value = tf.reduce_sum(critic_q_values*tf.one_hot(X_action,9**3),axis=1,keep_dims=True)
## Finding q value of the selected action value

learning_rate = 1e-2

y = tf.placeholder(tf.float32,shape=[None,1])
cost = tf.reduce_mean(tf.square(y-q_value))
global_step = tf.Variable(0,trainable=False,name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost,global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from collections import deque

replay_memory_size = 10000
replay_memory = deque([],maxlen=replay_memory_size)

def sample_memories(batch_size):
  indices = rnd.permutation(len(replay_memory))[:batch_size]
  cols = [[],[],[],[],[]]
  for idx in indices:
    memory = replay_memory[idx]
    for col, value in zip(cols, memory):
      col.append(value)
  cols = [np.array(col) for col in cols]
  return (cols[0],cols[1],cols[2].reshape(-1,1),cols[3],cols[4].reshape(-1,1))

eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000

def epsilon_greedy(q_values,step):
  ## Args
    ## q_values : get q values of each available actions
    ## step     : get the step count
  ## By epsilon greedy function it allows actor to go through the unchecked locations in the begining of the training
  ## Return
    ## Return which acton to do
  epsilon = max(eps_min,eps_max-(eps_max-eps_min)*step/eps_decay_steps)
  if np.random.rand()<epsilon:
    return np.random.randint(9**3)
  else:
    return np.argmax(q_values)

n_steps = 100000
training_start = 1000
training_interval = 3
save_steps = 50
copy_steps = 25
discount_rate = 0.95
skip_start = 0
batch_size = 50
iteration = 0
checkpoint_path = "./my_dqn.ckpt"
done = True

with tf.Session() as sess:
  if os.path.isfile(checkpoint_path):
    saver.restore(sess,checkpoint_path)
  else:
    init.run()
  while True:
    step = global_step.eval()
    if step >= n_steps:
      break
    iteration += 1
    if done:
      obs = env.reset()
      state = obs
    p1_q_values = actor_q_values.eval(feed_dict={X_state:[state]})
    p1_action = epsilon_greedy(p1_q_values,step)
    r = int(p1_action/81)       ## Get the Row to Add or Change
    c = int((p1_action%81)/9)   ## Get the Column to Add or Change
    n = ((p1_action%81)%9)+1    ## Get the number that is going to be in the selected box
    action = [r,c,n]
    print(action)
    obs,reward,done,info = env.step(action)
    next_state = obs
    
    replay_memory.append((state,p1_action,reward,next_state,1.0-done))
    state = next_state
    
    if iteration <training_start or iteration % training_interval != 0:
      continue
      
    X_state_val,X_action_val,rewards, X_next_state_val, continues = (sample_memories(batch_size))
    next_q_values = actor_q_values.eval(feed_dict={X_state:X_next_state_val})
    max_next_q_values = np.max(next_q_values,axis=1,keepdims=True)
    y_val = rewards+continues*discount_rate*max_next_q_values
    training_op.run(feed_dict={X_state:X_state_val,X_action:X_action_val,y:y_val})
    
    if step % copy_steps == 0:
      copy_critic_to_actor.run()
      
    if step % save_steps == 0:
      saver.save(sess,checkpoint_path)
