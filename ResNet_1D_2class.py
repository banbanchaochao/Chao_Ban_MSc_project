import tensorflow as tf
import numpy as np
import os
import random

BN_EPSILON = 0.001

keep_prob = 0.7

train_batch = 53

test_batch = 18

total_batch = 71

iter_num = 5300

label_1_1D_file = np.load('corticalThicknessData_1D_label_1.npz')
label_1_data = label_1_1D_file['Matrix']

label_3_1D_file = np.load('corticalThicknessData_1D_label_3.npz')
label_3_data = label_3_1D_file['Matrix']

label_for_train_and_test = np.zeros((8, 2))

label_for_train_and_test[0:4, 0] = 1

label_for_train_and_test[4:8, 1] = 1


def get_Conv1d(in_date, W):
    return tf.nn.conv2d(in_date, W, strides=[1, 1, 1, 1], padding='SAME')


def get_Conv1d_not_same(in_date, W):
    return tf.nn.conv2d(in_date, W, strides=[1, 1, 1, 1], padding='VALID')


def get_Conv1d_stride_2(in_date, W):
    return tf.nn.conv2d(in_date, W, strides=[1, 2, 1, 1], padding='SAME')


def get_Ave_pool(in_date, subtract, half):
    return tf.nn.avg_pool(in_date, ksize=[1, int(subtract), 1, 1], strides=[1, int(half), 1, 1], padding='VALID')


def get_Max_pool(in_date):
    return tf.nn.max_pool(in_date, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')


def get_batch_normalization(in_data):
    mean, variance = tf.nn.moments(in_data, axes=[0, 1, 2])
    dimension = in_data.get_shape().as_list()[-1]
    beta = tf.Variable(tf.zeros(dimension))
    gamma = tf.Variable(tf.ones(dimension))
    batch_normalization = tf.nn.batch_normalization(in_data, mean, variance, beta, gamma, BN_EPSILON)

    return batch_normalization


def get_LinearLayer(in_data, w, b):
    return tf.matmul(in_data, w) + b


def get_NonLinearLayer(in_data):
    return tf.nn.relu(in_data)


def get_Convolution_1(inputdata, w_1, b_1, w_2, b_2):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(conv_1, w_2) + b_2))
    conv_3 = get_Max_pool(conv_2)

    return conv_3

def get_Convolution_X(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(conv_1, w_2) + b_2))
    conv_3 = get_batch_normalization(get_Conv1d(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_2_pool(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(conv_1, w_2) + b_2))
    conv_3 = get_batch_normalization(get_Conv1d_not_same(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_X_pool(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(conv_1, w_2) + b_2))
    conv_3 = get_batch_normalization(get_Conv1d_stride_2(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_X_pool_not_same(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(conv_1, w_2) + b_2))
    conv_3 = get_batch_normalization(get_Conv1d_not_same(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_last_layer(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv1d(conv_1, w_2) + b_2))
    conv_3 = get_NonLinearLayer(get_batch_normalization(get_Conv1d_not_same(conv_2, w_3) + b_3))

    return conv_3


def get_Fusion_pool_add_channel(conv_input, conv_output, pad=0, add_channel=0):

    if pad == 0:
        conv_input_pool = conv_input
    elif pad == 1:
        conv_input_pool = get_Ave_pool(conv_input, 2, 2)
    elif pad == 2:
        conv_input_pool = get_Ave_pool(conv_input, 2, 1)
    else:
        conv_input_pool = get_Ave_pool(conv_input, 3, 1)

    if add_channel == 0:
        conv_next_input = get_NonLinearLayer(conv_input_pool + conv_output)
        # conv_next_input = get_NonLinearLayer(get_batch_normalization(conv_input_pool + conv_output))
    else:
        conv_input_2 = conv_input_pool
        conv_sum = tf.concat([conv_input_pool, conv_input_2], 3)
        conv_next_input = get_NonLinearLayer(conv_sum + conv_output)
        # conv_next_input = get_NonLinearLayer(get_batch_normalization(conv_sum + conv_output))

    return conv_next_input


def get_Convolutional_neural_network(x):

    # 1st strategy 1st block 163842 1 -> 81921 8
    W_conv1_1 = tf.Variable(tf.truncated_normal([3, 1, 1, 8], stddev=0.1))
    W_conv1_2 = tf.Variable(tf.truncated_normal([3, 1, 8, 8], stddev=0.1))

    b_conv1_1 = tf.Variable(tf.constant(0.1, shape=[8]))
    b_conv1_2 = tf.Variable(tf.constant(0.1, shape=[8]))

    # 2nd strategy 1st block 81921 8
    W_conv2_1_1 = tf.Variable(tf.truncated_normal([3, 1, 8, 8], stddev=0.1))
    W_conv2_1_2 = tf.Variable(tf.truncated_normal([3, 1, 8, 8], stddev=0.1))
    W_conv2_1_3 = tf.Variable(tf.truncated_normal([3, 1, 8, 8], stddev=0.1))

    b_conv2_1_1 = tf.Variable(tf.constant(0.1, shape=[8]))
    b_conv2_1_2 = tf.Variable(tf.constant(0.1, shape=[8]))
    b_conv2_1_3 = tf.Variable(tf.constant(0.1, shape=[8]))

    # 2nd strategy 2nd block 81921 8 -> 81920 16
    W_conv2_2_1 = tf.Variable(tf.truncated_normal([3, 1, 8, 8], stddev=0.1))
    W_conv2_2_2 = tf.Variable(tf.truncated_normal([3, 1, 8, 8], stddev=0.1))
    W_conv2_2_3 = tf.Variable(tf.truncated_normal([2, 1, 8, 16], stddev=0.1))

    b_conv2_2_1 = tf.Variable(tf.constant(0.1, shape=[8]))
    b_conv2_2_2 = tf.Variable(tf.constant(0.1, shape=[8]))
    b_conv2_2_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # 3rd strategy 1st block 81920 16
    W_conv3_1_1 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))
    W_conv3_1_2 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))
    W_conv3_1_3 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))

    b_conv3_1_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv3_1_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv3_1_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # 3rd strategy 2nd block 81920 16
    W_conv3_2_1 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))
    W_conv3_2_2 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))
    W_conv3_2_3 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))

    b_conv3_2_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv3_2_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv3_2_3 = tf.Variable(tf.constant(0.1, shape=[16]))

        # 3rd strategy 3rd block 81920 16 -> 40960 32
    W_conv3_3_1 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))
    W_conv3_3_2 = tf.Variable(tf.truncated_normal([3, 1, 16, 16], stddev=0.1))
    W_conv3_3_3 = tf.Variable(tf.truncated_normal([3, 1, 16, 32], stddev=0.1))

    b_conv3_3_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv3_3_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv3_3_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 4th strategy 1st block 40960 32
    W_conv4_1_1 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))
    W_conv4_1_2 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))
    W_conv4_1_3 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))

    b_conv4_1_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv4_1_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv4_1_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 4th strategy 2nd block 40960 32
    W_conv4_2_1 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))
    W_conv4_2_2 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))
    W_conv4_2_3 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))

    b_conv4_2_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv4_2_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv4_2_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 4th strategy 3rd block 40960 32 -> 20480 64
    W_conv4_3_1 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))
    W_conv4_3_2 = tf.Variable(tf.truncated_normal([3, 1, 32, 32], stddev=0.1))
    W_conv4_3_3 = tf.Variable(tf.truncated_normal([3, 1, 32, 64], stddev=0.1))

    b_conv4_3_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv4_3_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv4_3_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 5th strategy 1st block 20480 64
    W_conv5_1_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv5_1_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv5_1_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv5_1_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv5_1_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv5_1_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 5th strategy 2nd block 20480 64
    W_conv5_2_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv5_2_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv5_2_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv5_2_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv5_2_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv5_2_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 5th strategy 3rd block 20480 64 -> 10240 64
    W_conv5_3_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv5_3_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv5_3_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv5_3_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv5_3_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv5_3_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 6th strategy 1st block 10240 64
    W_conv6_1_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_1_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_1_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv6_1_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_1_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_1_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 6th strategy 2nd block 10240 64
    W_conv6_2_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_2_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_2_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv6_2_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_2_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_2_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 6th strategy 3rd block 10240 64
    W_conv6_3_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_3_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_3_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv6_3_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_3_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_3_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 6th strategy 4th block 10240 64 -> 5120 64
    W_conv6_4_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_4_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv6_4_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv6_4_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_4_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv6_4_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 7th strategy 1st block 5120 64
    W_conv7_1_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_1_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_1_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv7_1_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_1_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_1_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 7th strategy 2nd block 5120 64
    W_conv7_2_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_2_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_2_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv7_2_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_2_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_2_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 7th strategy 3rd block 5120 64
    W_conv7_3_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_3_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_3_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))

    b_conv7_3_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_3_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_3_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 7th strategy 4th block 5120 64 -> 2560 128
    W_conv7_4_1 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_4_2 = tf.Variable(tf.truncated_normal([3, 1, 64, 64], stddev=0.1))
    W_conv7_4_3 = tf.Variable(tf.truncated_normal([3, 1, 64, 128], stddev=0.1))

    b_conv7_4_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_4_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv7_4_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 8th strategy 1st block 2560 128
    W_conv8_1_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_1_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_1_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv8_1_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_1_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_1_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 8th strategy 2nd block 2560 128
    W_conv8_2_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_2_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_2_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv8_2_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_2_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_2_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 8th strategy 3rd block 2560 128
    W_conv8_3_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_3_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_3_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv8_3_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_3_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_3_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 8th strategy 4th block 2560 128
    W_conv8_4_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_4_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_4_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv8_4_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_4_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_4_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 8th strategy 5th block 2560 128 -> 1280 128
    W_conv8_5_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_5_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv8_5_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv8_5_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_5_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv8_5_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 9th strategy 1st block 1280 128
    W_conv9_1_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_1_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_1_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv9_1_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_1_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_1_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 9th strategy 2nd block 1280 128
    W_conv9_2_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_2_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_2_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv9_2_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_2_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_2_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 9th strategy 3rd block 1280 128
    W_conv9_3_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_3_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_3_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv9_3_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_3_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_3_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 9th strategy 4th block 1280 128
    W_conv9_4_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_4_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_4_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv9_4_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_4_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_4_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 9th strategy 5th block 1280 128 -> 640 128
    W_conv9_5_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_5_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv9_5_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv9_5_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_5_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv9_5_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 10th strategy 1st block 640 128
    W_conv10_1_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_1_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_1_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv10_1_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_1_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_1_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 10th strategy 2nd block 640 128
    W_conv10_2_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_2_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_2_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv10_2_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_2_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_2_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 10th strategy 3rd block 640 128
    W_conv10_3_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_3_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_3_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv10_3_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_3_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_3_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 10th strategy 4th block 640 128
    W_conv10_4_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_4_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_4_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))

    b_conv10_4_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_4_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_4_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 10th strategy 5th block 640 128 -> 320 256
    W_conv10_5_1 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_5_2 = tf.Variable(tf.truncated_normal([3, 1, 128, 128], stddev=0.1))
    W_conv10_5_3 = tf.Variable(tf.truncated_normal([3, 1, 128, 256], stddev=0.1))

    b_conv10_5_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_5_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv10_5_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 11th strategy 1st block 320 256
    W_conv11_1_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_1_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_1_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv11_1_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_1_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_1_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 11th strategy 2nd block 320 256
    W_conv11_2_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_2_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_2_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv11_2_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_2_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_2_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 11th strategy 3rd block 320 256
    W_conv11_3_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_3_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_3_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv11_3_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_3_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_3_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 11th strategy 4th block 320 256 -> 160 256
    W_conv11_4_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_4_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv11_4_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv11_4_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_4_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv11_4_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 12th strategy 1st block 160 256
    W_conv12_1_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_1_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_1_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv12_1_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_1_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_1_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 12th strategy 2nd block 160 256
    W_conv12_2_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_2_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_2_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv12_2_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_2_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_2_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 12th strategy 3rd block 160 256
    W_conv12_3_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_3_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_3_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv12_3_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_3_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_3_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 12th strategy 4th block 160 256 -> 80 256
    W_conv12_4_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_4_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv12_4_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv12_4_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_4_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv12_4_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 13th strategy 1st block 80 256
    W_conv13_1_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv13_1_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv13_1_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv13_1_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv13_1_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv13_1_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 13th strategy 2nd block 80 256
    W_conv13_2_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv13_2_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv13_2_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))

    b_conv13_2_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv13_2_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv13_2_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 13th strategy 3rd block 80 256 -> 40 512
    W_conv13_3_1 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv13_3_2 = tf.Variable(tf.truncated_normal([3, 1, 256, 256], stddev=0.1))
    W_conv13_3_3 = tf.Variable(tf.truncated_normal([3, 1, 256, 512], stddev=0.1))

    b_conv13_3_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv13_3_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv13_3_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 14th strategy 1st block 40 512
    W_conv14_1_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv14_1_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv14_1_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))

    b_conv14_1_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv14_1_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv14_1_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 14th strategy 2nd block 40 512
    W_conv14_2_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv14_2_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv14_2_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))

    b_conv14_2_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv14_2_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv14_2_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 14th strategy 3rd block 40 512 -> 20 512
    W_conv14_3_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv14_3_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv14_3_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))

    b_conv14_3_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv14_3_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv14_3_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 15th strategy 1st block 20 512
    W_conv15_1_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv15_1_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv15_1_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))

    b_conv15_1_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv15_1_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv15_1_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 15th strategy 2nd block 20 512
    W_conv15_2_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv15_2_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv15_2_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))

    b_conv15_2_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv15_2_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv15_2_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 15th strategy 3rd block 20 512 -> 10 512
    W_conv15_3_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv15_3_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv15_3_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))

    b_conv15_3_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv15_3_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv15_3_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 16th strategy 1st block 10 512
    W_conv16_1_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv16_1_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv16_1_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))

    b_conv16_1_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv16_1_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv16_1_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 16th strategy 2nd block 10 512 -> 5 1024
    W_conv16_2_1 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv16_2_2 = tf.Variable(tf.truncated_normal([3, 1, 512, 512], stddev=0.1))
    W_conv16_2_3 = tf.Variable(tf.truncated_normal([3, 1, 512, 1024], stddev=0.1))

    b_conv16_2_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv16_2_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv16_2_3 = tf.Variable(tf.constant(0.1, shape=[1024]))

    # 17th strategy 1st block 5 1024
    W_conv17_1_1 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))
    W_conv17_1_2 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))
    W_conv17_1_3 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))

    b_conv17_1_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    b_conv17_1_2 = tf.Variable(tf.constant(0.1, shape=[1024]))
    b_conv17_1_3 = tf.Variable(tf.constant(0.1, shape=[1024]))

    # 17th strategy 2nd block 5 2048 -> 3 1024
    W_conv17_2_1 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))
    W_conv17_2_2 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))
    W_conv17_2_3 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))

    b_conv17_2_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    b_conv17_2_2 = tf.Variable(tf.constant(0.1, shape=[1024]))
    b_conv17_2_3 = tf.Variable(tf.constant(0.1, shape=[1024]))

    # 18th strategy 1st block 5 1024 -> 1 2048
    W_conv18_1_1 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))
    W_conv18_1_2 = tf.Variable(tf.truncated_normal([3, 1, 1024, 1024], stddev=0.1))
    W_conv18_1_3 = tf.Variable(tf.truncated_normal([3, 1, 1024, 2048], stddev=0.1))

    b_conv18_1_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    b_conv18_1_2 = tf.Variable(tf.constant(0.1, shape=[1024]))
    b_conv18_1_3 = tf.Variable(tf.constant(0.1, shape=[2048]))

    x_CNN = tf.reshape(x, shape=[-1, 163842, 1, 1])

    # 81921 8
    conv_1_output = get_Convolution_1(x_CNN, W_conv1_1, b_conv1_1, W_conv1_2, b_conv1_2)

    conv_2_1_input = conv_1_output

    # 81921 8
    conv_2_1_output = get_Convolution_X(conv_2_1_input, W_conv2_1_1, b_conv2_1_1, W_conv2_1_2, b_conv2_1_2, W_conv2_1_3,
                                        b_conv2_1_3)

    conv_2_1_output = get_Fusion_pool_add_channel(conv_2_1_input, conv_2_1_output, pad=0, add_channel=0)

    conv_2_2_input = conv_2_1_output

    # 81920 16
    conv_2_2_output = get_Convolution_2_pool(conv_2_2_input, W_conv2_2_1, b_conv2_2_1, W_conv2_2_2, b_conv2_2_2,
                                             W_conv2_2_3, b_conv2_2_3)

    conv_2_2_output = get_Fusion_pool_add_channel(conv_2_2_input, conv_2_2_output, pad=2, add_channel=1)

    conv_3_1_input = conv_2_2_output

    # 81920 16
    conv_3_1_output = get_Convolution_X(conv_3_1_input, W_conv3_1_1, b_conv3_1_1, W_conv3_1_2, b_conv3_1_2, W_conv3_1_3,
                                        b_conv3_1_3)

    conv_3_1_output = get_Fusion_pool_add_channel(conv_3_1_input, conv_3_1_output, pad=0, add_channel=0)

    conv_3_2_input = conv_3_1_output

    # 81920 16
    conv_3_2_output = get_Convolution_X(conv_3_2_input, W_conv3_2_1, b_conv3_2_1, W_conv3_2_2, b_conv3_2_2, W_conv3_2_3,
                                        b_conv3_2_3)

    conv_3_2_output = get_Fusion_pool_add_channel(conv_3_2_input, conv_3_2_output, pad=0, add_channel=0)

    conv_3_3_input = conv_3_2_output

    # 40960 32
    conv_3_3_output = get_Convolution_X_pool(conv_3_3_input, W_conv3_3_1, b_conv3_3_1, W_conv3_3_2, b_conv3_3_2,
                                             W_conv3_3_3, b_conv3_3_3)

    conv_3_3_output = get_Fusion_pool_add_channel(conv_3_3_input, conv_3_3_output, pad=1, add_channel=1)

    conv_4_1_input = conv_3_3_output

    # 40960 32
    conv_4_1_output = get_Convolution_X(conv_4_1_input, W_conv4_1_1, b_conv4_1_1, W_conv4_1_2, b_conv4_1_2, W_conv4_1_3,
                                        b_conv4_1_3)

    conv_4_1_output = get_Fusion_pool_add_channel(conv_4_1_input, conv_4_1_output, pad=0, add_channel=0)

    conv_4_2_input = conv_4_1_output

    # 40960 32
    conv_4_2_output = get_Convolution_X(conv_4_2_input, W_conv4_2_1, b_conv4_2_1, W_conv4_2_2, b_conv4_2_2, W_conv4_2_3,
                                        b_conv4_2_3)

    conv_4_2_output = get_Fusion_pool_add_channel(conv_4_2_input, conv_4_2_output, pad=0, add_channel=0)

    conv_4_3_input = conv_4_2_output

    # 20480 64
    conv_4_3_output = get_Convolution_X_pool(conv_4_3_input, W_conv4_3_1, b_conv4_3_1, W_conv4_3_2, b_conv4_3_2,
                                             W_conv4_3_3, b_conv4_3_3)

    conv_4_3_output = get_Fusion_pool_add_channel(conv_4_3_input, conv_4_3_output,  pad=1, add_channel=1)

    conv_5_1_input = conv_4_3_output

    # 20480 64
    conv_5_1_output = get_Convolution_X(conv_5_1_input, W_conv5_1_1, b_conv5_1_1, W_conv5_1_2, b_conv5_1_2, W_conv5_1_3,
                                        b_conv5_1_3)

    conv_5_1_output = get_Fusion_pool_add_channel(conv_5_1_input, conv_5_1_output, pad=0, add_channel=0)

    conv_5_2_input = conv_5_1_output

    # 20480 64
    conv_5_2_output = get_Convolution_X(conv_5_2_input, W_conv5_2_1, b_conv5_2_1, W_conv5_2_2, b_conv5_2_2, W_conv5_2_3,
                                        b_conv5_2_3)

    conv_5_2_output = get_Fusion_pool_add_channel(conv_5_2_input, conv_5_2_output, pad=0, add_channel=0)

    conv_5_3_input = conv_5_2_output

    # 10240 64
    conv_5_3_output = get_Convolution_X_pool(conv_5_3_input, W_conv5_3_1, b_conv5_3_1, W_conv5_3_2, b_conv5_3_2,
                                             W_conv5_3_3, b_conv5_3_3)

    conv_5_3_output = get_Fusion_pool_add_channel(conv_5_3_input, conv_5_3_output, pad=1, add_channel=0)

    conv_6_1_input = conv_5_3_output

    # 10240 64
    conv_6_1_output = get_Convolution_X(conv_6_1_input, W_conv6_1_1, b_conv6_1_1, W_conv6_1_2, b_conv6_1_2, W_conv6_1_3,
                                        b_conv6_1_3)

    conv_6_1_output = get_Fusion_pool_add_channel(conv_6_1_input, conv_6_1_output, pad=0, add_channel=0)

    conv_6_2_input = conv_6_1_output

    # 10240 64
    conv_6_2_output = get_Convolution_X(conv_6_2_input, W_conv6_2_1, b_conv6_2_1, W_conv6_2_2, b_conv6_2_2, W_conv6_2_3,
                                        b_conv6_2_3)

    conv_6_2_output = get_Fusion_pool_add_channel(conv_6_2_input, conv_6_2_output, pad=0, add_channel=0)

    conv_6_3_input = conv_6_2_output

    # 10240 64
    conv_6_3_output = get_Convolution_X(conv_6_3_input, W_conv6_3_1, b_conv6_3_1, W_conv6_3_2, b_conv6_3_2, W_conv6_3_3,
                                        b_conv6_3_3)

    conv_6_3_output = get_Fusion_pool_add_channel(conv_6_3_input, conv_6_3_output, pad=0, add_channel=0)

    conv_6_4_input = conv_6_3_output

    # 5120 64
    conv_6_4_output = get_Convolution_X_pool(conv_6_4_input, W_conv6_4_1, b_conv6_4_1, W_conv6_4_2, b_conv6_4_2,
                                             W_conv6_4_3, b_conv6_4_3)

    conv_6_4_output = get_Fusion_pool_add_channel(conv_6_4_input, conv_6_4_output, pad=1, add_channel=0)

    conv_7_1_input = conv_6_4_output

    # 5120 64
    conv_7_1_output = get_Convolution_X(conv_7_1_input, W_conv7_1_1, b_conv7_1_1, W_conv7_1_2, b_conv7_1_2, W_conv7_1_3,
                                        b_conv7_1_3)

    conv_7_1_output = get_Fusion_pool_add_channel(conv_7_1_input, conv_7_1_output, pad=0, add_channel=0)

    conv_7_2_input = conv_7_1_output

    # 5120 64
    conv_7_2_output = get_Convolution_X(conv_7_2_input, W_conv7_2_1, b_conv7_2_1, W_conv7_2_2, b_conv7_2_2, W_conv7_2_3,
                                        b_conv7_2_3)

    conv_7_2_output = get_Fusion_pool_add_channel(conv_7_2_input, conv_7_2_output, pad=0, add_channel=0)

    conv_7_3_input = conv_7_2_output

    # 5120 64
    conv_7_3_output = get_Convolution_X(conv_7_3_input, W_conv7_3_1, b_conv7_3_1, W_conv7_3_2, b_conv7_3_2, W_conv7_3_3,
                                        b_conv7_3_3)

    conv_7_3_output = get_Fusion_pool_add_channel(conv_7_3_input, conv_7_3_output, pad=0, add_channel=0)

    conv_7_4_input = conv_7_3_output

    # 2560 128
    conv_7_4_output = get_Convolution_X_pool(conv_7_4_input, W_conv7_4_1, b_conv7_4_1, W_conv7_4_2, b_conv7_4_2,
                                             W_conv7_4_3, b_conv7_4_3)

    conv_7_4_output = get_Fusion_pool_add_channel(conv_7_4_input, conv_7_4_output, pad=1, add_channel=1)

    conv_8_1_input = conv_7_4_output

    # 2560 128
    conv_8_1_output = get_Convolution_X(conv_8_1_input, W_conv8_1_1, b_conv8_1_1, W_conv8_1_2, b_conv8_1_2, W_conv8_1_3,
                                        b_conv8_1_3)

    conv_8_1_output = get_Fusion_pool_add_channel(conv_8_1_input, conv_8_1_output, pad=0, add_channel=0)

    conv_8_2_input = conv_8_1_output

    # 2560 128
    conv_8_2_output = get_Convolution_X(conv_8_2_input, W_conv8_2_1, b_conv8_2_1, W_conv8_2_2, b_conv8_2_2, W_conv8_2_3,
                                        b_conv8_2_3)

    conv_8_2_output = get_Fusion_pool_add_channel(conv_8_2_input, conv_8_2_output, pad=0, add_channel=0)

    conv_8_3_input = conv_8_2_output

    # 2560 128
    conv_8_3_output = get_Convolution_X(conv_8_3_input, W_conv8_3_1, b_conv8_3_1, W_conv8_3_2, b_conv8_3_2, W_conv8_3_3,
                                        b_conv8_3_3)

    conv_8_3_output = get_Fusion_pool_add_channel(conv_8_3_input, conv_8_3_output, pad=0, add_channel=0)

    conv_8_4_input = conv_8_3_output

    # 2560 128
    conv_8_4_output = get_Convolution_X(conv_8_4_input, W_conv8_4_1, b_conv8_4_1, W_conv8_4_2, b_conv8_4_2, W_conv8_4_3,
                                        b_conv8_4_3)

    conv_8_4_output = get_Fusion_pool_add_channel(conv_8_4_input, conv_8_4_output, pad=0, add_channel=0)

    conv_8_5_input = conv_8_4_output

    # 1280 128
    conv_8_5_output = get_Convolution_X_pool(conv_8_5_input, W_conv8_5_1, b_conv8_5_1, W_conv8_5_2, b_conv8_5_2,
                                             W_conv8_5_3, b_conv8_5_3)

    conv_8_5_output = get_Fusion_pool_add_channel(conv_8_5_input, conv_8_5_output, pad=1, add_channel=0)

    conv_9_1_input = conv_8_5_output

    # 1280 128
    conv_9_1_output = get_Convolution_X(conv_9_1_input, W_conv9_1_1, b_conv9_1_1, W_conv9_1_2, b_conv9_1_2, W_conv9_1_3,
                                        b_conv9_1_3)

    conv_9_1_output = get_Fusion_pool_add_channel(conv_9_1_input, conv_9_1_output, pad=0, add_channel=0)

    conv_9_2_input = conv_9_1_output

    # 1280 128
    conv_9_2_output = get_Convolution_X(conv_9_2_input, W_conv9_2_1, b_conv9_2_1, W_conv9_2_2, b_conv9_2_2, W_conv9_2_3,
                                        b_conv9_2_3)

    conv_9_2_output = get_Fusion_pool_add_channel(conv_9_2_input, conv_9_2_output, pad=0, add_channel=0)

    conv_9_3_input = conv_9_2_output

    # 1280 128
    conv_9_3_output = get_Convolution_X(conv_9_3_input, W_conv9_3_1, b_conv9_3_1, W_conv9_3_2, b_conv9_3_2, W_conv9_3_3,
                                        b_conv9_3_3)

    conv_9_3_output = get_Fusion_pool_add_channel(conv_9_3_input, conv_9_3_output, pad=0, add_channel=0)

    conv_9_4_input = conv_9_3_output

    # 1280 128
    conv_9_4_output = get_Convolution_X(conv_9_4_input, W_conv9_4_1, b_conv9_4_1, W_conv9_4_2, b_conv9_4_2, W_conv9_4_3,
                                        b_conv9_4_3)

    conv_9_4_output = get_Fusion_pool_add_channel(conv_9_4_input, conv_9_4_output, pad=0, add_channel=0)

    conv_9_5_input = conv_9_4_output

    # 640 128
    conv_9_5_output = get_Convolution_X_pool(conv_9_5_input, W_conv9_5_1, b_conv9_5_1, W_conv9_5_2, b_conv9_5_2,
                                             W_conv9_5_3, b_conv9_5_3)

    conv_9_5_output = get_Fusion_pool_add_channel(conv_9_5_input, conv_9_5_output, pad=1, add_channel=0)

    conv_10_1_input = conv_9_5_output

    # 640 128
    conv_10_1_output = get_Convolution_X(conv_10_1_input, W_conv10_1_1, b_conv10_1_1, W_conv10_1_2, b_conv10_1_2,
                                         W_conv10_1_3, b_conv10_1_3)

    conv_10_1_output = get_Fusion_pool_add_channel(conv_10_1_input, conv_10_1_output, pad=0, add_channel=0)

    conv_10_2_input = conv_10_1_output

    # 640 128
    conv_10_2_output = get_Convolution_X(conv_10_2_input, W_conv10_2_1, b_conv10_2_1, W_conv10_2_2, b_conv10_2_2,
                                         W_conv10_2_3, b_conv10_2_3)

    conv_10_2_output = get_Fusion_pool_add_channel(conv_10_2_input, conv_10_2_output, pad=0, add_channel=0)

    conv_10_3_input = conv_10_2_output

    # 640 128
    conv_10_3_output = get_Convolution_X(conv_10_3_input, W_conv10_3_1, b_conv10_3_1, W_conv10_3_2, b_conv10_3_2,
                                         W_conv10_3_3, b_conv10_3_3)

    conv_10_3_output = get_Fusion_pool_add_channel(conv_10_3_input, conv_10_3_output, pad=0, add_channel=0)

    conv_10_4_input = conv_10_3_output

    # 640 128
    conv_10_4_output = get_Convolution_X(conv_10_4_input, W_conv10_4_1, b_conv10_4_1, W_conv10_4_2, b_conv10_4_2,
                                         W_conv10_4_3, b_conv10_4_3)

    conv_10_4_output = get_Fusion_pool_add_channel(conv_10_4_input, conv_10_4_output, pad=0, add_channel=0)

    conv_10_5_input = conv_10_4_output

    # 320 256
    conv_10_5_output = get_Convolution_X_pool(conv_10_5_input, W_conv10_5_1, b_conv10_5_1, W_conv10_5_2, b_conv10_5_2,
                                              W_conv10_5_3, b_conv10_5_3)

    conv_10_5_output = get_Fusion_pool_add_channel(conv_10_5_input, conv_10_5_output, pad=1, add_channel=1)

    conv_11_1_input = conv_10_5_output

    # 320 256
    conv_11_1_output = get_Convolution_X(conv_11_1_input, W_conv11_1_1, b_conv11_1_1, W_conv11_1_2, b_conv11_1_2,
                                         W_conv11_1_3, b_conv11_1_3)

    conv_11_1_output = get_Fusion_pool_add_channel(conv_11_1_input, conv_11_1_output, pad=0, add_channel=0)

    conv_11_2_input = conv_11_1_output

    # 320 256
    conv_11_2_output = get_Convolution_X(conv_11_2_input, W_conv11_2_1, b_conv11_2_1, W_conv11_2_2, b_conv11_2_2,
                                         W_conv11_2_3, b_conv11_2_3)

    conv_11_2_output = get_Fusion_pool_add_channel(conv_11_2_input, conv_11_2_output, pad=0, add_channel=0)

    conv_11_3_input = conv_11_2_output

    # 320 256
    conv_11_3_output = get_Convolution_X(conv_11_3_input, W_conv11_3_1, b_conv11_3_1, W_conv11_3_2, b_conv11_3_2,
                                         W_conv11_3_3, b_conv11_3_3)

    conv_11_3_output = get_Fusion_pool_add_channel(conv_11_3_input, conv_11_3_output, pad=0, add_channel=0)

    conv_11_4_input = conv_11_3_output

    # 160 256
    conv_11_4_output = get_Convolution_X_pool(conv_11_4_input, W_conv11_4_1, b_conv11_4_1, W_conv11_4_2, b_conv11_4_2,
                                              W_conv11_4_3, b_conv11_4_3)

    conv_11_4_output = get_Fusion_pool_add_channel(conv_11_4_input, conv_11_4_output, pad=1, add_channel=0)

    conv_12_1_input = conv_11_4_output

    # 160 256
    conv_12_1_output = get_Convolution_X(conv_12_1_input, W_conv12_1_1, b_conv12_1_1, W_conv12_1_2, b_conv12_1_2,
                                         W_conv12_1_3, b_conv12_1_3)

    conv_12_1_output = get_Fusion_pool_add_channel(conv_12_1_input, conv_12_1_output, pad=0, add_channel=0)

    conv_12_2_input = conv_12_1_output

    # 160 256
    conv_12_2_output = get_Convolution_X(conv_12_2_input, W_conv12_2_1, b_conv12_2_1, W_conv12_2_2, b_conv12_2_2,
                                         W_conv12_2_3, b_conv12_2_3)

    conv_12_2_output = get_Fusion_pool_add_channel(conv_12_2_input, conv_12_2_output, pad=0, add_channel=0)

    conv_12_3_input = conv_12_2_output

    # 160 256
    conv_12_3_output = get_Convolution_X(conv_12_3_input, W_conv12_3_1, b_conv12_3_1, W_conv12_3_2, b_conv12_3_2,
                                         W_conv12_3_3, b_conv12_3_3)

    conv_12_3_output = get_Fusion_pool_add_channel(conv_12_3_input, conv_12_3_output, pad=0, add_channel=0)

    conv_12_4_input = conv_12_3_output

    # 80 256
    conv_12_4_output = get_Convolution_X_pool(conv_12_4_input, W_conv12_4_1, b_conv12_4_1, W_conv12_4_2, b_conv12_4_2,
                                              W_conv12_4_3, b_conv12_4_3)

    conv_12_4_output = get_Fusion_pool_add_channel(conv_12_4_input, conv_12_4_output, pad=1, add_channel=0)

    conv_13_1_input = conv_12_4_output

    # 80 256
    conv_13_1_output = get_Convolution_X(conv_13_1_input, W_conv13_1_1, b_conv13_1_1, W_conv13_1_2, b_conv13_1_2,
                                         W_conv13_1_3, b_conv13_1_3)

    conv_13_1_output = get_Fusion_pool_add_channel(conv_13_1_input, conv_13_1_output, pad=0, add_channel=0)

    conv_13_2_input = conv_13_1_output

    # 80 256
    conv_13_2_output = get_Convolution_X(conv_13_2_input, W_conv13_2_1, b_conv13_2_1, W_conv13_2_2, b_conv13_2_2,
                                         W_conv13_2_3, b_conv13_2_3)

    conv_13_2_output = get_Fusion_pool_add_channel(conv_13_2_input, conv_13_2_output, pad=0, add_channel=0)

    conv_13_3_input = conv_13_2_output

    # 40 512
    conv_13_3_output = get_Convolution_X_pool(conv_13_3_input, W_conv13_3_1, b_conv13_3_1, W_conv13_3_2, b_conv13_3_2,
                                              W_conv13_3_3, b_conv13_3_3)

    conv_13_3_output = get_Fusion_pool_add_channel(conv_13_3_input, conv_13_3_output, pad=1, add_channel=1)

    conv_14_1_input = conv_13_3_output

    # 40 512
    conv_14_1_output = get_Convolution_X(conv_14_1_input, W_conv14_1_1, b_conv14_1_1, W_conv14_1_2, b_conv14_1_2,
                                         W_conv14_1_3, b_conv14_1_3)

    conv_14_1_output = get_Fusion_pool_add_channel(conv_14_1_input, conv_14_1_output, pad=0, add_channel=0)

    conv_14_2_input = conv_14_1_output

    # 40 512
    conv_14_2_output = get_Convolution_X(conv_14_2_input, W_conv14_2_1, b_conv14_2_1, W_conv14_2_2, b_conv14_2_2,
                                         W_conv14_2_3, b_conv14_2_3)

    conv_14_2_output = get_Fusion_pool_add_channel(conv_14_2_input, conv_14_2_output, pad=0, add_channel=0)

    conv_14_3_input = conv_14_2_output

    # 20 512
    conv_14_3_output = get_Convolution_X_pool(conv_14_3_input, W_conv14_3_1, b_conv14_3_1, W_conv14_3_2, b_conv14_3_2,
                                              W_conv14_3_3, b_conv14_3_3)

    conv_14_3_output = get_Fusion_pool_add_channel(conv_14_3_input, conv_14_3_output, pad=1, add_channel=0)

    conv_15_1_input = conv_14_3_output

    # 20 512
    conv_15_1_output = get_Convolution_X(conv_15_1_input, W_conv15_1_1, b_conv15_1_1, W_conv15_1_2, b_conv15_1_2,
                                         W_conv15_1_3, b_conv15_1_3)

    conv_15_1_output = get_Fusion_pool_add_channel(conv_15_1_input, conv_15_1_output, pad=0, add_channel=0)

    conv_15_2_input = conv_15_1_output

    # 20 512
    conv_15_2_output = get_Convolution_X(conv_15_2_input, W_conv15_2_1, b_conv15_2_1, W_conv15_2_2, b_conv15_2_2,
                                         W_conv15_2_3, b_conv15_2_3)

    conv_15_2_output = get_Fusion_pool_add_channel(conv_15_2_input, conv_15_2_output, pad=0, add_channel=0)

    conv_15_3_input = conv_15_2_output

    # 10 512
    conv_15_3_output = get_Convolution_X_pool(conv_15_3_input, W_conv15_3_1, b_conv15_3_1, W_conv15_3_2, b_conv15_3_2,
                                              W_conv15_3_3, b_conv15_3_3)

    conv_15_3_output = get_Fusion_pool_add_channel(conv_15_3_input, conv_15_3_output, pad=1, add_channel=0)

    conv_16_1_input = conv_15_3_output

    # 10 512
    conv_16_1_output = get_Convolution_X(conv_16_1_input, W_conv16_1_1, b_conv16_1_1, W_conv16_1_2, b_conv16_1_2,
                                         W_conv16_1_3, b_conv16_1_3)

    conv_16_1_output = get_Fusion_pool_add_channel(conv_16_1_input, conv_16_1_output, pad=0, add_channel=0)

    conv_16_2_input = conv_16_1_output

    # 5 1024
    conv_16_2_output = get_Convolution_X_pool(conv_16_2_input, W_conv16_2_1, b_conv16_2_1, W_conv16_2_2, b_conv16_2_2,
                                              W_conv16_2_3, b_conv16_2_3)

    conv_16_2_output = get_Fusion_pool_add_channel(conv_16_2_input, conv_16_2_output, pad=1, add_channel=1)

    conv_17_1_input = conv_16_2_output

    # 5 1024
    conv_17_1_output = get_Convolution_X(conv_17_1_input, W_conv17_1_1, b_conv17_1_1, W_conv17_1_2, b_conv17_1_2,
                                         W_conv17_1_3, b_conv17_1_3)

    conv_17_1_output = get_Fusion_pool_add_channel(conv_17_1_input, conv_17_1_output, pad=0, add_channel=0)

    conv_17_2_input = conv_17_1_output

    # 3 1024
    conv_17_2_output = get_Convolution_X_pool_not_same(conv_17_2_input, W_conv17_2_1, b_conv17_2_1, W_conv17_2_2,
                                                       b_conv17_2_2, W_conv17_2_3, b_conv17_2_3)

    conv_17_2_output = get_Fusion_pool_add_channel(conv_17_2_input, conv_17_2_output, pad=3, add_channel=0)

    conv_18_1_input = conv_17_2_output

    # 1 2048
    conv_18_1_output = get_Convolution_last_layer(conv_18_1_input, W_conv18_1_1, b_conv18_1_1, W_conv18_1_2,
                                                  b_conv18_1_2, W_conv18_1_3, b_conv18_1_3)

    conv_18_vect = tf.reshape(conv_18_1_output, [-1, 2048])

    W_fc = tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.1))
    W_linear = tf.Variable(tf.random_normal([2048, 2], stddev=0.1))

    b_fc = tf.Variable(tf.constant(0.1, shape=[2048]))
    b_linear = tf.Variable(tf.constant(0.1, shape=[2]))

    non_fc = get_NonLinearLayer(get_LinearLayer(conv_18_vect, W_fc, b_fc))

    non_fc = tf.nn.dropout(non_fc, keep_prob)

    L_out = get_LinearLayer(non_fc, W_linear, b_linear)

    return L_out


def save_model(session, ith_net):
    file_name = './1D_model_2class_' + str(int(ith_net)) + '/'
    document_name = file_name + 'model.checkpoint'
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    saver = tf.train.Saver()
    saver.save(session, document_name)


def main():

    X = tf.placeholder("float", [None, 163842])
    Y = tf.placeholder("float", [None, 2])
    LEARNING_RATE = tf.placeholder("float")

    model = get_Convolutional_neural_network(X)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimize = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    predict = tf.nn.softmax(model)
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    for ith_net in range(1, 101):

        label_1_id_test_new = random.sample(range(607), 72)
        label_3_id_test_new = random.sample(range(284), 72)

        label_1_id_all = []
        label_3_id_all = []
        for i in range(607):
            label_1_id_all.append(i)
        for i in range(284):
            label_3_id_all.append(i)

        for j in range(72):
            label_1_id_all.remove(label_1_id_test_new[j])
            label_3_id_all.remove(label_3_id_test_new[j])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Training started........")
            LR = 0.001
            total_loss = np.zeros((iter_num, 1))
            test_accuracy = np.zeros((100, 1))
            for indexIter in range(iter_num):

                batch_x = np.zeros((8, 163842))

                label_1_id_train_new = random.sample(range(535), 4)
                label_3_id_train_new = random.sample(range(212), 4)

                for k in range(4):
                    batch_x[k] = label_1_data[label_1_id_train_new[k]]
                    batch_x[k + 4] = label_3_data[label_3_id_train_new[k]]

                _, each_loss = sess.run([optimize, loss], feed_dict={X: batch_x, Y: label_for_train_and_test, LEARNING_RATE: LR})
                total_loss[indexIter] = each_loss
                print('ith_net', ith_net)
                print('indexIter %d: Loss %.5f' % (indexIter, each_loss))

                if (indexIter+1) % train_batch == 0:

                    index_acc = (indexIter+1) / train_batch

                    accuracy_test = 0
                    for test_indexIter in range(18):

                        testX = np.zeros((8, 163842))
                        lower_index_test = int(test_indexIter * 4)
                        upper_index_test = int((test_indexIter + 1) * 4)

                        label_1_id_test = label_1_id_test_new[lower_index_test:upper_index_test]
                        label_3_id_test = label_3_id_test_new[lower_index_test:upper_index_test]

                        for k in range(4):
                            testX[k] = label_1_data[label_1_id_test[k]]
                            testX[k + 4] = label_3_data[label_3_id_test[k]]

                        accuracy_test_t, pp = sess.run([accuracy, predict], feed_dict={X: testX, Y: label_for_train_and_test})
                        ppv = np.argmax(pp, 1)
                        print(test_indexIter)
                        print(ppv)
                        accuracy_test += accuracy_test_t

                    accuracy_test /= test_batch
                    error_test = 1 - accuracy_test

                    print('Iteration %d: Accuracy %.5f Error %.5f (test)' % (indexIter, accuracy_test, error_test))

                    test_accuracy[int(index_acc-1)] = accuracy_test

                if indexIter >= 2650:
                    LR = 0.001 * (1 - (indexIter - 2650)/iter_num)

            save_model(sess, ith_net)

            document_name = '1D_2class_curve_' + str(int(ith_net)) + '.npz'

            np.savez(document_name,
                     total_loss=total_loss,
                     test_accuracy=test_accuracy
                     )

if __name__ == '__main__':
    main()
