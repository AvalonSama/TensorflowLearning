from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

mnist = input_data.read_data_sets("MNIST_data/",reshape = True, one_hot = False)
train_data = np.reshape(mnist.train.images,[-1,28,28])
train_lable = mnist.train.labels
test_data = np.reshape(mnist.test.images,[-1,28,28])
test_lable = mnist.test.labels

class CNN(object):
    def __init__(self,batch_size,image_height,image_weight,hidden_size,class_num,learning_rate,max_turn):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_weight = image_weight
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.max_turn = max_turn
        self.x = tf.placeholder(dtype = tf.float32, shape=[None,self.image_height,self.image_weight])
        self.y = tf.placeholder(dtype = tf.int64, shape = [None])


        with tf.name_scope("conv1"): #定义卷积核1
            self.conv1_weight = tf.get_variable(
                name = "conv1_weight",
                initializer= tf.random_uniform_initializer(-1,1),
                dtype= tf.float32,
                shape= [5,5,1,32]
            )
            self.conv1_bias = tf.get_variable(
                name = "conv1_bias",
                initializer=tf.random_uniform_initializer(-1,1),
                dtype= tf.float32,
                shape = [32]
            )

        with tf.name_scope("conv2"): #定义卷积核2
            self.conv2_weight = tf.get_variable(
                name = "conv2_weight",
                initializer= tf.random_uniform_initializer(-1,1),
                dtype= tf.float32,
                shape= [5,5,32,64]
            )
            self.conv2_bias = tf.get_variable(
                name = "conv2_bias",
                initializer=tf.random_uniform_initializer(-1,1),
                dtype= tf.float32,
                shape = [64]
            )
        
        
        with tf.name_scope("softmax"):
            self.softmax_weight = tf.get_variable(
                name = "softmax_weight",
                initializer = tf.random_uniform_initializer(-1,1),
                dtype= tf.float32,
                shape= [7*7*64,self.class_num]
            )
            self.softmax_bias = tf.get_variable(
                name = "softmax_bias",
                initializer = tf.random_uniform_initializer(-1,1),
                dtype = tf.float32,
                shape = [self.class_num]
            )
    

    def cnn(self,inputs):
        batchsize = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs,[-1,28,28,1])
        conv1 = tf.nn.conv2d(inputs,self.conv1_weight,[1,1,1,1],padding = 'SAME')+self.conv1_bias
        a_conv1 = tf.nn.tanh(conv1)
        p_conv1 = tf.nn.max_pool(a_conv1,[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

        conv2 = tf.nn.conv2d(p_conv1,self.conv2_weight,[1,1,1,1],padding = 'SAME')+self.conv2_bias
        a_conv2 = tf.nn.tanh(conv2)
        p_conv2 = tf.nn.max_pool(a_conv2,[1,2,2,1],[1,2,2,1],padding = 'SAME')

        temp = tf.reshape(p_conv2,[batchsize,-1])
        predict = tf.matmul(temp,self.softmax_weight)+self.softmax_bias
        return predict
    
    
    def getBatch(self):
        size = np.shape(train_data)[0]
        for i in range(0,size,self.batch_size):
            lower = i
            uper = min(size,i+self.batch_size)
            yield train_data[lower:uper],train_lable[lower:uper]

    def run(self):
        predict = self.cnn(self.x)

        with tf.name_scope('cost'):
            one_hot_lable = tf.one_hot(self.y,10)
            cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = one_hot_lable))
        
        with tf.name_scope('train'):
            global_step = tf.Variable(0,name = 'step',trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost,global_step=global_step)
        
        with tf.name_scope('acc'):
            bool_count = tf.equal(tf.argmax(predict,1),self.y)
            count = tf.reduce_sum(tf.cast(bool_count,tf.int32))
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(self.max_turn):
                batchGeter = self.getBatch()
                loss = 0
                step = 0
                while True:
                    try:
                        train_data,train_lable = next(batchGeter)
                    except:
                        break
                    feeddic ={
                        self.x:train_data,
                        self.y:train_lable
                    }
                    _,temp_loss,temp_step = sess.run([optimizer,cost,global_step],feeddic)
                    loss+=temp_loss
                    step = temp_step
                    feeddic ={
                        self.x:test_data,
                        self.y:test_lable
                    }
                    acc = 1.0*sess.run(count,feeddic)/np.shape(test_data)[0]
                print("step: {}     cost: {}      acc: {}".format(step,loss, acc))



if __name__ == "__main__":
    BATCHSIZE = 500
    IMAGE_HEIGHT = 28
    IMAGE_WEIGHT = 28
    HIDDEN_SIZE = 100
    CLASS_NUM = 10
    LEARNING_RATE = 0.1
    MAX_TURN = 20


    cnn = CNN(
        batch_size = BATCHSIZE,
        image_height = IMAGE_HEIGHT,
        image_weight = IMAGE_WEIGHT,
        hidden_size = HIDDEN_SIZE,
        class_num = CLASS_NUM,
        learning_rate = LEARNING_RATE,
        max_turn = MAX_TURN
    )
    cnn.run()
