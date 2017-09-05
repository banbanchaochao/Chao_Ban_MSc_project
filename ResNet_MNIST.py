import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

BN_EPSILON = 0.001



def get_Conv2d(in_date, W):
    return tf.nn.conv2d(in_date, W, strides=[1, 1, 1, 1], padding='SAME')


def get_Conv2d_not_same(in_date, W):
    return tf.nn.conv2d(in_date, W, strides=[1, 1, 1, 1], padding='VALID')


def get_Conv2d_stride_2(in_date, W):
    return tf.nn.conv2d(in_date, W, strides=[1, 2, 2, 1], padding='SAME')


def get_Max_pool(in_date):
    return tf.nn.max_pool(in_date, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_Ave_pool(in_date):
    return tf.nn.avg_pool(in_date, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_Ave_pool_not_same(in_date):
    return tf.nn.avg_pool(in_date, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')


def get_LinearLayer(in_data, w, b):
    return tf.matmul(in_data, w) + b


def get_NonLinearLayer(in_data):
    return tf.nn.relu(in_data)


def get_batch_normalization(in_data):
    mean, variance = tf.nn.moments(in_data, axes=[0, 1, 2])
    dimension = in_data.get_shape().as_list()[-1]
    beta = tf.Variable(tf.zeros(dimension))
    gamma = tf.Variable(tf.ones(dimension))
    batch_normalization = tf.nn.batch_normalization(in_data, mean, variance, beta, gamma, BN_EPSILON)

    return batch_normalization


def get_Convolution_1(inputdata, w_1, b_1, w_2):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv2d_stride_2(conv_1, w_2)))

    return conv_2


def get_Convolution_X(inputdata, w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_1, w_2) + b_2))
    conv_3 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_2, w_3) + b_3))
    # conv_4 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_3, w_4) + b_4))
    conv_4 = get_batch_normalization(get_Conv2d(conv_3, w_4) + b_4)

    return conv_4


def get_Convolution_X_pool(inputdata, w_1, b_1, w_2, b_2, w_3, b_3, w_4):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_1, w_2) + b_2))
    conv_3 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_2, w_3) + b_3))
    # conv_4 = get_NonLinearLayer(get_batch_normalization(get_Conv2d_stride_2(conv_3, w_4)))
    conv_4 = get_batch_normalization(get_Conv2d_stride_2(conv_3, w_4))

    return conv_4


def get_Convolution_X_not_normal_pool(inputdata, w_1, b_1, w_2, b_2, w_3, b_3, w_4):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_1, w_2) + b_2))
    conv_3 = get_NonLinearLayer(get_batch_normalization(get_Conv2d_not_same(conv_2, w_3) + b_3))
    # conv_4 = get_NonLinearLayer(get_batch_normalization(get_Conv2d_stride_2(conv_3, w_4)))
    conv_4 = get_batch_normalization(get_Conv2d_stride_2(conv_3, w_4))

    return conv_4


def get_Convolution_X_last_layer(inputdata, w_1, b_1, w_2, b_2, w_3, b_3, w_4):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_1, w_2) + b_2))
    conv_3 = get_NonLinearLayer(get_batch_normalization(get_Conv2d(conv_2, w_3) + b_3))
    conv_4 = get_NonLinearLayer(get_batch_normalization(get_Conv2d_not_same(conv_3, w_4)))

    return conv_4


def get_Fusion(conv_input, conv_output):

    conv_next_input = get_NonLinearLayer(conv_input + conv_output)

    return conv_next_input


def get_Fusion_add_channel(conv_input, num, conv_output):

    # conv_input_padding = tf.pad(conv_input, [[0, 0], [0, 0], [0, 0], [0, num]], "CONSTANT")
    conv_input_2 = conv_input
    conv_sum = tf.concat([conv_input, conv_input_2], 3)
    # conv_next_input = get_NonLinearLayer(conv_input_padding + conv_output)
    conv_next_input = get_NonLinearLayer(conv_sum + conv_output)

    return conv_next_input


def get_Fusion_pool(conv_input, conv_output, pad=0):

    if pad == 0:
        conv_input_pool = get_Ave_pool(conv_input)
    else:
        conv_input_pool = get_Ave_pool(get_Ave_pool_not_same(conv_input))
    conv_next_input = get_NonLinearLayer(conv_input_pool + conv_output)

    return conv_next_input


def get_Fusion_pool_add_channel(conv_input, num, conv_output, pad=0):

    if pad == 0:
        conv_input_pool = get_Ave_pool(conv_input)
    else:
        conv_input_pool = get_Ave_pool(get_Ave_pool_not_same(conv_input))
    # conv_input_padding = tf.pad(conv_input_pool, [[0, 0], [0, 0], [0, 0], [0, num]], "CONSTANT")
    conv_input_2 = conv_input
    conv_sum = tf.concat([conv_input, conv_input_2], 3)
    # conv_next_input = get_NonLinearLayer(conv_input_padding + conv_output)
    conv_next_input = get_NonLinearLayer(conv_sum + conv_output)

    return conv_next_input


def get_Convolutional_neural_network(x):

    # 1st strategy 1st block 28 -> 14
    W_1_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
    W_1_2 = tf.Variable(tf.truncated_normal([3, 3, 16, 16], stddev=0.1))
    b_1_1 = tf.Variable(tf.constant(0.1, shape=[16]))

    # 2nd strategy 1st block
    W_conv2_1_1 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
    W_conv2_1_2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_1_3 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_1_4 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))

    b_conv2_1_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_1_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_1_3 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_1_4 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 2nd strategy 2nd block
    W_conv2_2_1 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_2_2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_2_3 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_2_4 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))

    b_conv2_2_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_2_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_2_3 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_2_4 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 2nd strategy 3rd block 14 -> 7
    W_conv2_3_1 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_3_2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_3_3 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    W_conv2_3_4 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))

    b_conv2_3_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_3_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2_3_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 3nd strategy 1st block
    W_conv3_1_1 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    W_conv3_1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_1_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_1_4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))

    b_conv3_1_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_1_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_1_3 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_1_4 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 3nd strategy 2nd block
    W_conv3_2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_2_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_2_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_2_4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))

    b_conv3_2_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_2_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_2_3 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_2_4 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 3nd strategy 3rd block
    W_conv3_3_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_3_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_3_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_3_4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))

    b_conv3_3_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_3_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_3_3 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_3_4 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 3nd strategy 4th block
    W_conv3_4_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_4_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_4_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_4_4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))

    b_conv3_4_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_4_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_4_3 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_4_4 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 3nd strategy 5th block
    W_conv3_5_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_5_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_5_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_5_4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))

    b_conv3_5_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_5_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_5_3 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_5_4 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 3nd strategy 6th block 7 -> 3
    W_conv3_6_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_6_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_6_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    W_conv3_6_4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))

    b_conv3_6_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_6_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3_6_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 4nd strategy 1st block
    W_conv4_1_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
    W_conv4_1_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_1_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_1_4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))

    b_conv4_1_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_1_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_1_3 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_1_4 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 4nd strategy 2nd block
    W_conv4_2_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_2_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_2_4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))

    b_conv4_2_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_2_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_2_3 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_2_4 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 4nd strategy 3rd block
    W_conv4_3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_3_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_3_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
    W_conv4_3_4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))

    b_conv4_3_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_3_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv4_3_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv_1_output = get_Convolution_1(x, W_1_1, b_1_1, W_1_2)

    conv_2_1_input = conv_1_output

    conv_2_1_output = get_Convolution_X(conv_2_1_input, W_conv2_1_1, b_conv2_1_1, W_conv2_1_2, b_conv2_1_2, W_conv2_1_3,
                                        b_conv2_1_3, W_conv2_1_4, b_conv2_1_4)

    conv_2_1_output = get_Fusion_add_channel(conv_2_1_input, 16, conv_2_1_output)

    conv_2_2_input = conv_2_1_output

    conv_2_2_output = get_Convolution_X(conv_2_2_input, W_conv2_2_1, b_conv2_2_1, W_conv2_2_2, b_conv2_2_2, W_conv2_2_3,
                                        b_conv2_2_3, W_conv2_2_4, b_conv2_2_4)

    conv_2_2_output = get_Fusion(conv_2_2_input, conv_2_2_output)

    conv_2_3_input = conv_2_2_output

    conv_2_3_output = get_Convolution_X_pool(conv_2_3_input, W_conv2_3_1, b_conv2_3_1, W_conv2_3_2, b_conv2_3_2,
                                             W_conv2_3_3, b_conv2_3_3, W_conv2_3_4)

    conv_2_3_output = get_Fusion_pool(conv_2_3_input, conv_2_3_output, pad=0)

    conv_3_1_input = conv_2_3_output

    conv_3_1_output = get_Convolution_X(conv_3_1_input, W_conv3_1_1, b_conv3_1_1, W_conv3_1_2, b_conv3_1_2, W_conv3_1_3,
                                        b_conv3_1_3, W_conv3_1_4, b_conv3_1_4)

    conv_3_1_output = get_Fusion_add_channel(conv_3_1_input, 32, conv_3_1_output)

    conv_3_2_input = conv_3_1_output

    conv_3_2_output = get_Convolution_X(conv_3_2_input, W_conv3_2_1, b_conv3_2_1, W_conv3_2_2, b_conv3_2_2, W_conv3_2_3,
                                        b_conv3_2_3, W_conv3_2_4, b_conv3_2_4)

    conv_3_2_output = get_Fusion(conv_3_2_input, conv_3_2_output)

    conv_3_3_input = conv_3_2_output

    conv_3_3_output = get_Convolution_X(conv_3_3_input, W_conv3_3_1, b_conv3_3_1, W_conv3_3_2, b_conv3_3_2, W_conv3_3_3,
                                        b_conv3_3_3, W_conv3_3_4, b_conv3_3_4)

    conv_3_3_output = get_Fusion(conv_3_3_input, conv_3_3_output)

    conv_3_4_input = conv_3_3_output

    conv_3_4_output = get_Convolution_X(conv_3_4_input, W_conv3_4_1, b_conv3_4_1, W_conv3_4_2, b_conv3_4_2, W_conv3_4_3,
                                        b_conv3_4_3, W_conv3_4_4, b_conv3_4_4)

    conv_3_4_output = get_Fusion(conv_3_4_input, conv_3_4_output)

    conv_3_5_input = conv_3_4_output

    conv_3_5_output = get_Convolution_X(conv_3_5_input, W_conv3_5_1, b_conv3_5_1, W_conv3_5_2, b_conv3_5_2, W_conv3_5_3,
                                        b_conv3_5_3, W_conv3_5_4, b_conv3_5_4)

    conv_3_5_output = get_Fusion(conv_3_5_input, conv_3_5_output)

    conv_3_6_input = conv_3_5_output

    conv_3_6_output = get_Convolution_X_not_normal_pool(conv_3_6_input, W_conv3_6_1, b_conv3_6_1, W_conv3_6_2,
                                                        b_conv3_6_2, W_conv3_6_3, b_conv3_6_3, W_conv3_6_4)

    conv_3_6_output = get_Fusion_pool(conv_3_6_input, conv_3_6_output, pad=1)

    conv_4_1_input = conv_3_6_output

    conv_4_1_output = get_Convolution_X(conv_4_1_input, W_conv4_1_1, b_conv4_1_1, W_conv4_1_2, b_conv4_1_2, W_conv4_1_3,
                                        b_conv4_1_3, W_conv4_1_4, b_conv4_1_4)

    conv_4_1_output = get_Fusion_add_channel(conv_4_1_input, 64, conv_4_1_output)

    conv_4_2_input = conv_4_1_output

    conv_4_2_output = get_Convolution_X(conv_4_2_input, W_conv4_2_1, b_conv4_2_1, W_conv4_2_2, b_conv4_2_2, W_conv4_2_3,
                                        b_conv4_2_3, W_conv4_2_4, b_conv4_2_4)

    conv_4_2_output = get_Fusion(conv_4_2_input, conv_4_2_output)

    conv_4_3_input = conv_4_2_output

    conv_4_3_output = get_Convolution_X_last_layer(conv_4_3_input, W_conv4_3_1, b_conv4_3_1, W_conv4_3_2, b_conv4_3_2,
                                                   W_conv4_3_3, b_conv4_3_3, W_conv4_3_4)

    conv_4_3_vect = tf.reshape(conv_4_3_output, [-1, 1 * 1 * 128])

    W_fc = tf.Variable(tf.truncated_normal([128, 1024], stddev=0.1))
    b_fc = tf.Variable(tf.constant(0.1, shape=[1024]))

    W_linear = tf.Variable(tf.random_normal([1024, 10], stddev=0.1))
    b_linear = tf.Variable(tf.constant(0.1, shape=[10]))

    fc = get_LinearLayer(conv_4_3_vect, W_fc, b_fc)
    non_fc = get_NonLinearLayer(fc)

    # keep_prob = 0.8

    # non_fc = tf.nn.dropout(non_fc, keep_prob)

    L_out = get_LinearLayer(non_fc, W_linear, b_linear)

    return L_out


def main():

    print("Loading the data......")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    print("Finished: data loaded. Stats below: ")

    (nrTrainSamples, dimX) = trainX.shape
    (nrTestSamples, dimY) = testY.shape

    X = tf.placeholder("float", [None, dimX])
    Y = tf.placeholder("float", [None, dimY])
    LEARNING_RATE = tf.placeholder("float")

    model = get_Convolutional_neural_network(X)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimize = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    predict = tf.nn.softmax(model)
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    train_acc_save = np.zeros((100, 1))
    test_acc_save = np.zeros((100, 1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training started........")
        LR = 0.008
        for indexIter in range(1, 100001):

            batch_x, batch_y = mnist.train.next_batch(100)
            sess.run(optimize, feed_dict={X: batch_x, Y: batch_y, LEARNING_RATE: LR})
            if indexIter % 1000 == 0:

                index_int = int(indexIter / 1000)

                accuracy_train = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                error_train = 1 - accuracy_train
                accuracy_test = sess.run(accuracy, feed_dict={X: testX, Y: testY})
                error_test = 1 - accuracy_test

                train_acc_save[index_int] = accuracy_train
                test_acc_save[index_int] = accuracy_test

                print('Iteration %d: Accuracy %.5f(training) %.5f(testing)' % (indexIter, accuracy_train, accuracy_test))
                print('Iteration %d: Error %.5f(training) %.5f(testing)' % (indexIter, error_train, error_test))
                print("LEARNING_RATE : ", LR)

            if indexIter > 50000:
                LR = 0.008 * (1 - (indexIter - 50000) / 100000)

    np.savez('mnist_deep_fuse',
             train_acc_save=train_acc_save,
             test_acc_save=test_acc_save
             )

if __name__ == '__main__':
    main()
