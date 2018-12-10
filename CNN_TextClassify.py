from tools.DataLoader import *
import tensorflow as tf
import numpy as np


class CNN_TestClassification(object):
    def __init__(self,
                 embedding_path,
                 train_data_path,
                 test_data_path,
                 batch_size,
                 hidden_size,
                 class_num,
                 learning_rate,
                 max_turn
                 ):
        self.embedding_path = embedding_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size

        self.word_map, self.embedding, self.embedding_dimention = Loadembedding(self.embedding_path)
        self.train_data, self.train_label,max_sen_len = LoadText(self.train_data_path, self.word_map)
        self.max_sen_len = max_sen_len
        self.test_data, self.text_label,max_sen_len = LoadText(self.test_data_path, self.word_map)
        self.max_sen_len = max(self.max_sen_len,max_sen_len)
        self.train_data = PADDING(self.train_data,self.max_sen_len)
        self.test_data = PADDING(self.test_data,self.max_sen_len)

        self.hidden_size = hidden_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.max_turn = max_turn

        self.x = tf.placeholder(tf.int32, [None, self.max_sen_len])
        self.y = tf.placeholder(tf.int32, [None])

        def getVariable(inshape, name):
            return tf.get_variable(
                name=name,
                initializer=tf.random_uniform_initializer(-1, 1),
                dtype=tf.float32,
                shape=inshape
            )
        with tf.name_scope("conv"):
            self.conv1_weight = getVariable(
                [5, self.embedding_dimention, 1, self.hidden_size], "conv1")
            self.conv2_weight = getVariable(
                [4, self.embedding_dimention, 1, self.hidden_size], "conv2")
            self.conv3_weight = getVariable(
                [3, self.embedding_dimention, 1, self.hidden_size], "conv3")

            self.conv1_bias = getVariable([self.hidden_size], "conv1_bias")
            self.conv2_bias = getVariable([self.hidden_size], "conv2_bias")
            self.conv3_bias = getVariable([self.hidden_size], "conv3_bias")

        with tf.name_scope("softmax"):
            self.softmax_weight = getVariable(
                [3*self.hidden_size, self.class_num], "softmax_weight")
            self.softmax_bias = getVariable([self.class_num], "softmax_bias")

    def cnn(self, inputs):
        inputs = tf.reshape(
            inputs, [-1, self.max_sen_len, self.embedding_dimention, 1])

        conv_hidden1 = tf.nn.conv2d(inputs, self.conv1_weight, [
                                    1, 1, 1, 1], padding='VALID')+self.conv1_bias
        conv_hidden2 = tf.nn.conv2d(inputs, self.conv2_weight, [
                                    1, 1, 1, 1], padding='VALID')+self.conv3_bias
        conv_hidden3 = tf.nn.conv2d(inputs, self.conv3_weight, [
                                    1, 1, 1, 1], padding='VALID')+self.conv3_bias

        a_hidden1 = tf.nn.relu(conv_hidden1)
        a_hidden2 = tf.nn.relu(conv_hidden2)
        a_hidden3 = tf.nn.relu(conv_hidden3)

        hidden1 = tf.nn.max_pool(
            a_hidden1, [1, self.max_sen_len-4, 1, 1], [1, 1, 1, 1], padding='VALID')
        hidden2 = tf.nn.max_pool(
            a_hidden2, [1, self.max_sen_len-3, 1, 1], [1, 1, 1, 1], padding='VALID')
        hidden3 = tf.nn.max_pool(
            a_hidden3, [1, self.max_sen_len-2, 1, 1], [1, 1, 1, 1], padding='VALID')

        hidden1 = tf.reshape(hidden1,[-1,self.hidden_size])
        hidden2 = tf.reshape(hidden2,[-1,self.hidden_size])
        hidden3 = tf.reshape(hidden3,[-1,self.hidden_size])
        combine_hidden = tf.concat((hidden1, hidden2, hidden3), 1)

        output = tf.matmul(
            combine_hidden, self.softmax_weight)+self.softmax_bias
        return output

    def getBatch(self):
        size = np.shape(self.train_data)[0]
        for i in range(0, size, self.batch_size):
            lower = i
            upper = min(i+self.batch_size, size)
            yield self.train_data[lower:upper], self.train_label[lower:upper]

    def run(self):
        inputs = tf.nn.embedding_lookup(self.embedding, self.x)
        predict = self.cnn(inputs)

        with tf.name_scope("loss"):
            one_hot_lable = tf.one_hot(self.y, self.class_num)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=predict, labels=one_hot_lable))

        with tf.name_scope("train"):
            global_step = tf.Variable(0, trainable=False, name='step')
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.max_turn):
                batchGeter = self.getBatch()
                total_cost = 0
                now_step = 0
                while True:
                    try:
                        data, label = next(batchGeter)
                    except:
                        break
                    batchsize = np.shape(data)[0]
                    data = np.reshape(data,[batchsize,self.max_sen_len])
                    feeddic = {
                        self.x: data,
                        self.y: label
                    }
                    _, cost, step = sess.run([optimizer, loss, global_step], feed_dict=feeddic)
                    total_cost += cost
                    now_step = step
                print("turn:  {}   step:  {}   cost:  {}".format(
                    i, now_step, total_cost))


if __name__ == "__main__":
    EMBEDDING_PATH = "./word_embeddings/sst-Google-vectors.txt"
    TRAIN_DATA_PATH = "./TextClassification_data/SST/binary/sst.binary.train1.txt"
    TEST_DATA_PATH = "./TextClassification_data/SST/binary/sst.binary.test1.txt"
    BATCH_SIZE = 100
    HIDDEN_SIZE = 100
    CLASS_NUM = 2
    LEARNING_RATE = 0.1
    MAX_TURN = 20

    cnn = CNN_TestClassification(
        embedding_path = EMBEDDING_PATH,
        train_data_path = TRAIN_DATA_PATH,
        test_data_path = TEST_DATA_PATH,
        batch_size = BATCH_SIZE,
        hidden_size = HIDDEN_SIZE,
        class_num = CLASS_NUM,
        learning_rate = LEARNING_RATE,
        max_turn = MAX_TURN
    )
    cnn.run()
