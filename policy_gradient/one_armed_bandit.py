import tensorflow as tf
import numpy as np

#List out our bandits
bandits = [0.2,0,-0.2,-5]
num_bandits = len(bandits)

def pullBandit(bandit):
    #get a random number
    result = np.random.randn(1)

    if(result)>bandit:
        #return a positive reward
        return 1
    else:
        #return a negative reward
        return -1

tf.reset_default_graph()

#establish the feed-forward part of network, to do choosing
#initialized to 1's just to make agent optimistic, per tutorial
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights,0)

#this part establishes training procedure.
#Feed in reward and chosen action into network, compute loss, and use it to update the network
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

#gonna slice one val from weights that corresponds to index specified by 'action holder' placeholder
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

#set number of eps to train agent
total_episodes = 1000
#scoreboard for the bandits!
total_reward = np.zeros(num_bandits)

e = 0.1 # gotta set eps

init = tf.global_variables_initializer()

#tf.summary.scalar("action_holder",action_holder)
#tf.summary.scalar("reward_holder",reward_holder)
#tf.summary.scalar("responsible_weight",responsible_weight)
#tf.summary.scalar("loss",loss)

logs_path = '/tf_logs'


#Launch tensorflow graph
with tf.Session() as sess:

    writer = tf.summary.FileWriter(logs_path,sess.graph)
    sess.run(init)
    i = 0
    while i < total_episodes:

        #choose between random or greedy action
        if np.random.rand(1)<e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)

        reward = pullBandit(bandits[action]) #get reward from selected action

        #update the network
        _,resp, ww = sess.run([update,responsible_weight,weights],feed_dict={reward_holder:[reward],action_holder:[action]})

        #writer.add_summary(summary,i)
        #update running tally
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_bandits) + "bandits: " + str(total_reward))
        i+=1

writer.close()
print("the agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising...")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print ("...and it was right!")
else:
    print ("...and it was wrong!")
