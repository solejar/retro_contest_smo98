import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as pyplot

#cute little line from the github code to make it python3 and python2 agnostic
try:
    xrange = xrange
except:
    xrange = range

#this is future reward discount
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0,r.size)):
        running_add = running_add*gamma+r[t] #recursively multiply by gamma the running add
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self,lr,s_size,a_size,h_size):
        #These lines establish the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training procedure. we feed the reward and chosen action into the networkself
        #to compute the loss, and use it to update the networkself.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(self.output,[-1]),self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate (tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

max_eps = 999

#little helper function
def run_episode(env,agent=None,render=False):

    s = env.reset()
    running_reward = 0

    for _ in range(max_eps):
        if render:
            env.render()
        if agent is None:
            action = env.action_space.sample()
        else:
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist==a)

            s, reward, done, _ = env.step(a)
            running_reward += reward

        if done:
            break
    return total_reward

if __name__=='__main__':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    total_episodes = 5000
    update_frequency = 5

    tf.reset_default_graph()

    myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent

    init = tf.global_variables_initializer()

    #Launch graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_length = []

        #this is an  length 2 array with a[0] as the [4,8] array of weights inited for the first connection layer
        #and a[1] as the [8,2] array of weights connecting the second layer
        gradBuffer = sess.run(tf.trainable_variables())

        #set to 0 while preserving shape
        for idx,grad in enumerate(gradBuffer):
            gradBuffer[idx] = grad*0

        while i< total_episodes:
            s = env.reset()
            running_reward = 0
            ep_history = []
            for j in range(max_eps):
                #probabilistically pick an action
                a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in: [s]})

                a = np.random.choice(a_dist[0],p=a_dist[0])
                #this line is retrieving the corresponding index which is actually the action
                a = np.argmax(a_dist==a)

                s1,r,d,_ = env.step(a) #get the reward for taking the action
                #tack onto the epoch history
                ep_history.append([s,a,r,s1])

                #move the state forward
                s = s1

                #add the reward for the episode
                running_reward += r

                #if we're done (game over)
                if d == True:
                    #update the network
                    ep_history = np.array(ep_history)
                    ep_history[:,2] = discount_rewards(ep_history[:,2])
                    #The vstack forces state in as a n [1,1] arrays, for some reason that's necessary while action and reward
                    #can handle [n,1] array inputs?
                    feed_dict = {myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                    grads = sess.run(myAgent.gradients, feed_dict = feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad
                    if i % update_frequency == 0  and i != 0:
                        feed_dict = dictionary = dict(zip(myAgent.gradient_holders,gradBuffer))
                        _ = sess.run(myAgent.update_batch,feed_dict = feed_dict)
                        for ix, grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad*0

                    total_reward.append(running_reward)
                    total_length.append(j)
                    break

                #update running tally of scores
            if i % 100 == 0:
                print(np.mean(total_reward[-100:]))
            i+=1
        final_reward = run_episode(env,agent=myAgent,render=True)
        print("And final reward for final policy is %f",np.mean(total_reward[-100:]))
