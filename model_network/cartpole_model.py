import numpy as np
import _pickle as cPickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

#from modelAny import *

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
#from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

import gym

env = gym.make('CartPole-v0')

#Hyperparameters
H = 8 # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99 #future discount
decay_rate = 0.99 # decay factory for RMSprop leaky sum of grad^2
resume = False #resume from previous checkpoint?

model_bs = 3 #batch size when learning from model
real_bs = 3 #batch size when learning from real environment

#model initialization
D= 4 #input dimensionality

#state to action, this is a policy network
tf.reset_default_graph()
observations = tf.placeholder(tf.float32,[None,4],name="input_x")
W1 = tf.get_variable("W1",shape=[4,H],initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)

#[w1,w2]
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1],name="input_y")

#The more sure you were were about your choice, the closer this var gets to 0. The less sure you were, the greater the magnitude
loglik = tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))
advantages = tf.placeholder(tf.float32,name="reward_signal")

#Some scenarios, to explain loss model:
#1) sure, good advantages = small positive number, small priority to increase (good rewards here, but surety indicates lack of confidence in ability to improve)
#2) not sure, good advantages = high positive number, high priority to increase weight quickly (good rewards here, and lack of surety indicates a good opportunity to improve)
#3) sure, bad advantages = small negative number, small priority to decrease, want weight to go down slowly (bad rewards, but also not much opportunity to improve here)
#4) not sure, bad advantages = high negative number, high priority decrease, want weight to go down quickly (lack of surety indicates good opportunity, reduce weight to strengthen inverse correlation)
loss = -tf.reduce_mean(loglik*advantages)
newGrads = tf.gradients(loss,tvars)

#then feed it back in
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

mH = 256 # model layer a_size
#model network, takes in an [x1 x2 x3 x4 | a] [1,5] array, returns the predicted next state, reward, and done
input_data = tf.placeholder(tf.float32,[None,5])
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w",[mH,50])
    softmax_b = tf.get_variable("softmax_b",[50])

previous_state = tf.placeholder(tf.float32,[None,5],name="previous_state")
W1M = tf.get_variable("W1M",shape=[5,mH],initializer=tf.contrib.layers.xavier_initializer())

#2 hidden layers here
B1M = tf.Variable(tf.zeros([mH]),name="B1M")
layer1M = tf.nn.relu(tf.matmul(previous_state,W1M)+B1M)
W2M = tf.get_variable("W2M",shape=[mH,mH],initializer=tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([mH]),name="B2M")
layer2M = tf.nn.relu(tf.matmul(layer1M,W2M)+B2M)
w0 = tf.get_variable("w0",shape=[mH,4],initializer=tf.contrib.layers.xavier_initializer())
wR = tf.get_variable("wR",shape=[mH,1],initializer=tf.contrib.layers.xavier_initializer())
wD = tf.get_variable("wD",shape=[mH,1],initializer=tf.contrib.layers.xavier_initializer())

b0 = tf.Variable(tf.zeros([4]),name="b0")
bR = tf.Variable(tf.zeros([1]),name="bR")
bD = tf.Variable(tf.ones([1]),name="bD")

#3 output nodes
predicted_observation =tf.matmul(layer2M,w0,name="predicted_observation") + b0
predicted_reward = tf.matmul(layer2M,wR,name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M,wD,name="predicted_done") + bD)

true_observation = tf.placeholder(tf.float32,[None,4],name="true_observation")
true_reward = tf.placeholder(tf.float32,[None,1],name="true_done")
true_done = tf.placeholder(tf.float32,[None,1],name="true_done")

predicted_state = tf.concat([predicted_observation,predicted_reward,predicted_done],1)

observation_loss = tf.square(true_observation- predicted_observation)

reward_loss = tf.square(true_reward-predicted_reward)

#calculate loss, each multiply handles true_done being 1, and 0 respectively
done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done)
#i think he's using log here because the difference between 0 and 1 is really big here, but it doesn't scale appropriately for least squares
#Yeah this loss model is pretty much only for [0,1]
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

modelAdam = tf.train.AdamOptimizer(learning_rate = learning_rate)
updateModel = modelAdam.minimize(model_loss)

def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad* 0
    return gradBuffer

#discounted future rewards for policy gradient descent
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0,r.size)):
        running_add = running_add*gamma+r[t]
        discounted_r[t] = running_add
    return discounted_r

#function that steps the model given an action
def stepModel(sess,xs,action):
    #array is [x1 x2 x3 x4 | action] for most recent action
    toFeed = np.reshape(np.hstack([xs[-1][0],np.array(action)]),[1,5])
    #get prediction from model
    myPredict = sess.run([predicted_state],feed_dict={previous_state: toFeed})
    #predicted reward for action
    reward = myPredict[0][:,4]
    #predicted observation for next state
    observation = myPredict[0][:,0:4]

    #clip the prediction to the physical limits of the environment (?)
    observation[:,0] = np.clip(observation[:,0],-2.4,2.4)
    observation[:,2] = np.clip(observation[:,2],-0.4,0.4)

    #turn prediction of done into binary form
    doneP = np.clip(myPredict[0][:,5],0,1)
    if doneP>0.1 or len(xs)>=300:
        done = True
    else:
        done = False
    return observation, reward, done

#xs is array of observations reshaped to be (1,4)
#ys is array of complements of action
#ds is array of dones recorded
#and drs is reward history
xs,drs,ys,ds = [],[],[],[]

running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs

#it eithers trains the model from the env,
#or it trains the policy from the model
#configs are: 0,1,0 (shown here) and 1,0,1
drawFromModel = False
trainTheModel = True
trainThePolicy = False

#basically the episode where the last batch ended, to see if we have collected a full new minibatch yet
switch_point = 1

#Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    #we only want to train 5000 episodes
    while episode_number <= 5000:
        #start displaying environment once performance is acceptably high
        if (reward_sum/batch_size >150 and drawFromModel == False) or rendering == True:
            env.render()
            rendering = True
        x = np.reshape(observation,[1,4])

        tfprob = sess.run(probability,feed_dict={observations: x})
        action = 1 if np.random.uniform() <tfprob else 0

        #record various intermediates (needed later for backprop)
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        #step the model or real environment
        if drawFromModel==False:
            observation,reward,done,info = env.step(action)
        else:
            observation,reward,done = stepModel(sess,xs,action)
        reward_sum += reward

        ds.append(done*1)
        drs.append(reward) #record reward (has to be done after we call step() to get reward for previous action)

        if done:
            if drawFromModel == False:
                real_episodes += 1
            episode_number += 1
            #stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs,drs,ys,ds = [],[],[],[] # reset array memory
            if trainTheModel == True:
                #splices the last array val of actions
                actions = np.array([np.abs(y-1) for y in epy][:-1])
                #ignore the last row as well
                state_prevs = epx[:-1,:]
                state_prevs = np.hstack([state_prevs,actions])
                #last set of observations, for the 'new' state
                state_nexts = epx[1:,:]
                rewards = np.array(epr[1:,:])
                dones = np.array(epd[1:,:])
                state_nextsAll = np.hstack([state_nexts,rewards,dones])

                feed_dict={previous_state: state_prevs, true_observation: state_nexts, true_done: dones, true_reward: rewards}
                loss,pStates,_ = sess.run([model_loss,predicted_state,updateModel],feed_dict)

            if trainThePolicy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                #center to 0 and scale to 1
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(newGrads, feed_dict = {observations: epx,input_y: epy, advantages: discounted_epr})

                #if gradients become too large, end training process
                if np.sum(tGrad[0]==tGrad[0])==0:
                    break
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

            if switch_point +batch_size == episode_number:
                switch_point = episode_number
                if trainThePolicy == True:
                    sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum*0.01
                if drawFromModel ==False:
                    print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' %(real_episodes,reward_sum/real_bs,action,running_reward/real_bs))
                    if reward_sum/batch_size>200:
                        break
                reward_sum = 0

                #once the model has been trained on 100 episodes, we alternate between training on model and real environment
                if episode_number >100:
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy
            if drawFromModel== True:
                observation = np.random.uniform(-0.1,0.1,[4]) #Generate reasonable starting point
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = real_bs

print(real_episodes)
