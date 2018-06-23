# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:05:51 2018

@author: Silesh Chandran
"""

import tensorflow as tf

xin=tf.placeholder(tf.float32,shape=[4,2],name="input")
yout=tf.placeholder(tf.float32,shape=[4,1],name="output")

w1=tf.Variable(tf.random_uniform([2,2],-1,1),name="weight1")
w2=tf.Variable(tf.random_uniform([2,1],-1,1),name="weight2")

b1=tf.Variable(tf.zeros([2]),name="Bias1")
b2=tf.Variable(tf.zeros([1]),name="Bias2")


a2=tf.sigmoid(tf.matmul(xin,w1)+b1)
out=tf.sigmoid(tf.matmul(a2,w2)+b2)

cost=tf.reduce_mean(( (yout*tf.log(out))+((1-yout)*tf.log(1.0-out)))*-1)
train_step=tf.train.MomentumOptimizer(0.01,0.9).minimize(cost)

X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[0],[1],[1],[0]]

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

w=tf.summary.FileWriter('./logs',sess.graph)
w.close()

for i in range(50000):
    sess.run(train_step,feed_dict={xin:X,yout:Y})
    if i % 10000 == 0:
        print('Epoch ', i)
        print('output ', sess.run(out, feed_dict={xin:X,yout:Y}))
        print('Weight1', sess.run(w1))
        print('Bias1 ', sess.run(b1))
        print('Weight2 ', sess.run(w2))
        print('Bias2 ', sess.run(b2))
        print('cost ', sess.run(cost, feed_dict={xin:X,yout:Y}))