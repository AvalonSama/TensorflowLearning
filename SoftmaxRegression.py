from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/",reshape = True, one_hot = False)
train_data = mnist.train.images
train_lable = mnist.train.labels
test_data = mnist.test.images
test_lable = mnist.test.labels

class SoftmaxRegression(object):
    def __init__(self,batch_size,class_num,input_dimention,maxturn,learning_rate):
        self.batch_size = batch_size
        self.class_num = class_num
        self.input_dimention = input_dimention
        self.maxturn = maxturn
        self.learning_rate = learning_rate

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32,[None, self.input_dimention])
            self.y = tf.placeholder(tf.int64,[None])
        
        with tf.name_scope('variables'):
            self.weights = tf.get_variable(
                name = "weights",
                shape = [self.input_dimention,self.class_num],
                initializer= tf.random_uniform_initializer(-1.0,1.0),
                dtype= tf.float32
            )
            self.bias = tf.get_variable(
                name = "biase",
                dtype = tf.float32,
                initializer= tf.random_uniform_initializer(-1.0,1.0),
                shape = [self.class_num]
            )
    
    def getBatch(self):
        size = np.shape(train_data)[0]
        for i in range(0,size,self.batch_size):
            lower = i
            uper = min(size,i+self.batch_size)
            yield train_data[lower:uper],train_lable[lower:uper]
    
    def softmaxRegression(self,inputs):
        outputs = tf.matmul(inputs,self.weights)+self.bias
        return outputs
    
    def run(self):
        predict = self.softmaxRegression(self.x)

        with tf.name_scope('loss'):
            one_hot_lable = tf.one_hot(self.y,10)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = one_hot_lable))
        
        with tf.name_scope('train'):
            global_step = tf.Variable(0,name = "step",trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,global_step=global_step)

        with tf.name_scope('acc'):
            correct_num = tf.equal(tf.argmax(predict,1),self.y)
            correct_num = tf.reduce_sum(tf.cast(correct_num,tf.int32))
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for i in range(self.maxturn):
                batchGeter = self.getBatch()
                all_cost = 0
                cnt = 0
                while True:
                    cnt+=1
                    try:
                        inputs_data,inputs_lable = next(batchGeter)
                    except:
                        break
                    feeddic = {
                        self.x : inputs_data,
                        self.y : inputs_lable
                    }
                    _, step ,cost = sess.run([optimizer,global_step,loss],feed_dict=feeddic)
                    num = np.shape(inputs_data)[0]
                    all_cost+=cost*num
                feeddic = {
                    self.x:test_data,
                    self.y:test_lable
                }
                count = sess.run([correct_num],feed_dict=feeddic)
                acc = 1.0*count[0]/np.shape(test_data)[0]
                print("step:{}    cost:{}     acc:{:.6f}".format(step,all_cost,acc))

if __name__ == "__main__":

    BATCH_SIZE = 5000
    CLASS_NUM = 10
    INPUT_DIMENTION = np.shape(train_data)[1]
    MAXTURN = 100
    LEARNING_RATE = 0.01

    sr = SoftmaxRegression(
        batch_size = BATCH_SIZE,
        class_num = CLASS_NUM,
        input_dimention = INPUT_DIMENTION,
        maxturn = MAXTURN,
        learning_rate = LEARNING_RATE
    )
    sr.run()

