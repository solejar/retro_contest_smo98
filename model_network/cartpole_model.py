import numpy as np
import _pickle as cPickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

import gym

def model_layer(input,input_channels, output_channels, name="model_layer", activation=None,use_biases=True):
    with tf.variable_scope(name):
        w = tf.get_variable("w",shape=[input_channels,output_channels],initializer=tf.contrib.layers.xavier_initializer())
        if use_biases:
            b = tf.Variable(tf.zeros([output_channels]),name="b")
            tf.summary.histogram("biases",b)

            z = tf.matmul(input,w) + b

        else:
            z = tf.matmul(input,w)

        if (activation is not None):
            act = activation(z)
        else:
            act = z

        tf.summary.histogram("weights",w)
        tf.summary.histogram("activations",act)

        return act

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

if __name__ =='__main__':

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

    tf.reset_default_graph()

    #state to action, this is a policy network
    with tf.variable_scope("policy_network"):

        observations = tf.placeholder(tf.float32,[None,4],name="input_x")
        layer1 = model_layer(observations,4,H,name="hidden_layer1",activation=tf.nn.relu,use_biases=False)
        probability = model_layer(layer1,H,1,name="output_layer", activation=tf.nn.sigmoid,use_biases=False)

        #[w1,w2]
        tvars = tf.trainable_variables()

        with tf.variable_scope("loss"):
            input_y = tf.placeholder(tf.float32,[None,1],name="input_y")

            #The more sure you were were about your choice, the closer this var gets to 0. The less sure you were, the greater the magnitude
            with tf.variable_scope("loglik"):
                loglik = tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))

            advantages = tf.placeholder(tf.float32,name="reward_signal")

            #Some scenarios, to explain loss model:
            #1) sure, good advantages = small positive number, small priority to increase (good rewards here, but surety indicates lack of confidence in ability to improve)
            #2) not sure, good advantages = high positive number, high priority to increase weight quickly (good rewards here, and lack of surety indicates a good opportunity to improve)
            #3) sure, bad advantages = small negative number, small priority to decrease, want weight to go down slowly (bad rewards, but also not much opportunity to improve here)
            #4) not sure, bad advantages = high negative number, high priority decrease, want weight to go down quickly (lack of surety indicates good opportunity, reduce weight to strengthen inverse correlation)

            loss = -tf.reduce_mean(loglik*advantages)

            policy_loss_summary = tf.summary.scalar("policy_loss",loss)

        with tf.variable_scope("gradients"):
            newGrads = tf.gradients(loss,tvars)

            #then feed it back in

        with tf.variable_scope("update_model"):
            W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
            W2Grad = tf.placeholder(tf.float32,name="batch_grad2")

            batchGrad = [W1Grad,W2Grad]
            #batchGrad = []
            adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
            updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

    #this is the next state model
    with tf.variable_scope("next_state_model"):

        mH = 256 # model layer a_size

        #model network, takes in an [x1 x2 x3 x4 | a] [1,5] array, returns the predicted next state, reward, and done
        previous_state = tf.placeholder(tf.float32,[None,5],name="previous_state")

        layer1M = model_layer(previous_state,5,mH,name="hidden_layer1",activation=tf.nn.relu)
        layer2M = model_layer(layer1M,mH,mH,name="hidden_layer2",activation=tf.nn.relu)

        predicted_observation = model_layer(layer2M,mH,4,name="predicted_observation")
        predicted_reward = model_layer(layer2M,mH,1,name="predicted_reward")
        predicted_done = model_layer(layer2M,mH,1,name="predicted_done",activation=tf.sigmoid)

        with tf.variable_scope("model_loss"):

            predicted_state = tf.concat([predicted_observation,predicted_reward,predicted_done],1)

            with tf.variable_scope("observation_loss"):
                true_observation = tf.placeholder(tf.float32,[None,4],name="true_observation")
                observation_loss = tf.square(true_observation- predicted_observation)

            with tf.variable_scope("reward_loss"):
                true_reward = tf.placeholder(tf.float32,[None,1],name="true_done")
                reward_loss = tf.square(true_reward-predicted_reward)

            #calculate loss, each multiply handles true_done being 1, and 0 respectively
            with tf.variable_scope("done_loss"):
                true_done = tf.placeholder(tf.float32,[None,1],name="true_done")
                #
                done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done)
                #i think he's using log here because the difference between 0 and 1 is really big here, but it doesn't scale appropriately for least squares
                #Yeah this loss model is pretty much only for [0,1]
                done_loss = -tf.log(done_loss)

            model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)
            model_loss_summary = tf.summary.scalar("model_loss",model_loss)

        with tf.variable_scope("update_model"):
            modelAdam = tf.train.AdamOptimizer(learning_rate = learning_rate)
            updateModel = modelAdam.minimize(model_loss)

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

    #Launch the session
    merged_policy_summary = tf.summary.merge([policy_loss_summary])
    merged_model_summary = tf.summary.merge([model_loss_summary])

    env = gym.make('CartPole-v0')
    with tf.Session() as sess:
        #this is a relative directory to where you call it
        writer = tf.summary.FileWriter('./tf_logs/cartpole_model')
        writer.add_graph(sess.graph)

        rendering = False
        sess.run(init)
        observation = env.reset()

        gradBuffer = sess.run(tvars)
        gradBuffer = resetGradBuffer(gradBuffer)

        #we only want to train 5000 episodes
        max_episodes = 2500
        while episode_number <= max_episodes:

            #start displaying environment once performance is acceptably high
            if (reward_sum/batch_size >180 and drawFromModel == False) or rendering == True:
            #if max_episodes >= (max_episodes)
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
                    if episode_number % 5 ==0:
                        summary,loss,pStates,_ = sess.run([merged_model_summary,model_loss,predicted_state,updateModel],feed_dict)
                        writer.add_summary(summary,episode_number)
                    else:
                        loss,pStates,_ = sess.run([model_loss,predicted_state,updateModel],feed_dict)

                if trainThePolicy == True:
                    discounted_epr = discount_rewards(epr).astype('float32')
                    #center to 0 and scale to 1
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)
                    if episode_number % 25 == 0:
                        summary, tGrad = sess.run([merged_policy_summary, newGrads], feed_dict = {observations: epx,input_y: epy, advantages: discounted_epr})
                        writer.add_summary(summary,episode_number)
                    else:
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
