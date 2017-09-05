import tensorflow as tf
import numpy as np
import os
import random

BN_EPSILON = 0.001

keep_prob = 0.7

train_batch = 71

test_batch = 23

total_batch = 94

iter_num = 7100

label_1_3D_file = np.load('corticalThicknessMatrix_label_1.npz')
label_1_data = label_1_3D_file['Matrix']

label_3_3D_file = np.load('corticalThicknessMatrix_label_3.npz')
label_3_data = label_3_3D_file['Matrix']

label_for_train_and_test = np.zeros((6, 2))

label_for_train_and_test[0:3, 0] = 1

label_for_train_and_test[3:6, 1] = 1


def get_Conv3d(in_date, W):
    return tf.nn.conv3d(in_date, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def get_Conv3d_not_same(in_date, W):
    return tf.nn.conv3d(in_date, W, strides=[1, 1, 1, 1, 1], padding='VALID')


def get_Conv3d_stride_2(in_date, W):
    return tf.nn.conv3d(in_date, W, strides=[1, 2, 2, 2, 1], padding='SAME')


def get_Max_pool(in_date):
    return tf.nn.max_pool3d(in_date, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def get_Ave_pool(in_date, size):
    return tf.nn.avg_pool3d(in_date, ksize=size, strides=[1, 2, 2, 2, 1], padding='SAME')


def get_Ave_pool_not_same(in_date, size):
    return tf.nn.avg_pool3d(in_date, ksize=size, strides=[1, 1, 1, 1, 1], padding='VALID')


def get_LinearLayer(in_data, w, b):
    return tf.matmul(in_data, w) + b


def get_NonLinearLayer(in_data):
    return tf.nn.relu(in_data)


def get_batch_normalization(in_data):
    mean, variance = tf.nn.moments(in_data, axes=[0, 1, 2, 3])
    dimension = in_data.get_shape().as_list()[-1]
    beta = tf.Variable(tf.zeros(dimension))
    gamma = tf.Variable(tf.ones(dimension))
    batch_normalization = tf.nn.batch_normalization(in_data, mean, variance, beta, gamma, BN_EPSILON)

    return batch_normalization


def get_Convolution_1(inputdata, w_1, b_1):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(inputdata, w_1) + b_1))
    conv_2 = get_Max_pool(conv_1)

    return conv_2


def get_Convolution_2(inputdata, w_1, b_1, w_2, b_2, w_3, b_3, block=1):

    if block == 1:
        conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d_not_same(inputdata, w_1) + b_1))
    else:
        conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(conv_1, w_2) + b_2))
    if block == 1:
        conv_3 = get_batch_normalization(get_Conv3d_stride_2(conv_2, w_3) + b_3)
    else:
        conv_3 = get_batch_normalization(get_Conv3d(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_3(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(conv_1, w_2) + b_2))
    conv_3 = get_batch_normalization(get_Conv3d(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_3_pool(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(conv_1, w_2) + b_2))
    conv_3 = get_batch_normalization(get_Conv3d_stride_2(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_X(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(conv_1, w_2) + b_2))
    conv_3 = get_batch_normalization(get_Conv3d(conv_2, w_3) + b_3)

    return conv_3


def get_Convolution_X_pool(inputdata, w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv3d_not_same(conv_1, w_2) + b_2))
    conv_3 = get_NonLinearLayer(get_batch_normalization(get_Conv3d_stride_2(conv_2, w_3) + b_3))
    conv_4 = get_batch_normalization(get_Conv3d(conv_3, w_4) + b_4)

    return conv_4


def get_Convolution_X_last_layer(inputdata, w_1, b_1, w_2, b_2, w_3, b_3):

    conv_1 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(inputdata, w_1) + b_1))
    conv_2 = get_NonLinearLayer(get_batch_normalization(get_Conv3d_not_same(conv_1, w_2) + b_2))
    conv_3 = get_NonLinearLayer(get_batch_normalization(get_Conv3d(conv_2, w_3) + b_3))

    return conv_3


def get_Fusion_pool_add_channel(conv_input, conv_output, size_same, size_not_same, pad=0, add_channel=0):

    if pad == 0:
        conv_input_pool = conv_input
    elif pad == 1:
        conv_input_pool = get_Ave_pool(conv_input, size_same)
    else:
        conv_input_pool = get_Ave_pool(get_Ave_pool_not_same(conv_input, size_not_same), size_same)

    if add_channel == 0:
        conv_next_input = get_NonLinearLayer(conv_input_pool + conv_output)
        # conv_next_input = get_NonLinearLayer(get_batch_normalization(conv_input_pool + conv_output))
    else:
        conv_input_2 = conv_input_pool
        conv_sum = tf.concat([conv_input_pool, conv_input_2], 4)
        conv_next_input = get_NonLinearLayer(conv_sum + conv_output)
        # conv_next_input = get_NonLinearLayer(get_batch_normalization(conv_sum + conv_output))

    return conv_next_input


def get_Convolutional_neural_network(x):

    # 1st strategy 1st block 84 218 146 1 -> 42 109 73 8
    W_conv1_1 = tf.Variable(tf.truncated_normal([7, 7, 7, 1, 8], stddev=0.1))

    b_conv1_1 = tf.Variable(tf.constant(0.1, shape=[8]))

    # 2nd strategy 1st block 42 109 73 16
    W_conv2_1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 8, 16], stddev=0.1))
    W_conv2_1_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_1_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))

    b_conv2_1_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_1_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_1_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # # 2nd strategy 2nd block 42 109 73 16
    # W_conv2_2_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    # W_conv2_2_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    # W_conv2_2_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    #
    # b_conv2_2_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    # b_conv2_2_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    # b_conv2_2_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # # 2nd strategy 3rd block 42 109 73 16
    # W_conv2_3_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    # W_conv2_3_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    # W_conv2_3_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    #
    # b_conv2_3_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    # b_conv2_3_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    # b_conv2_3_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # 2nd strategy 4th block 42 109 73 16
    W_conv2_4_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_4_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_4_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))

    b_conv2_4_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_4_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_4_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # 2nd strategy 5th block 42 109 73 16
    W_conv2_5_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_5_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_5_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))

    b_conv2_5_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_5_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_5_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # 2nd strategy 6th block 42 109 73 16
    W_conv2_6_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_6_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_6_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))

    b_conv2_6_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_6_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_6_3 = tf.Variable(tf.constant(0.1, shape=[16]))

    # 2nd strategy 7th block 42 109 73 16 -> 20 54 46 32
    W_conv2_7_1 = tf.Variable(tf.truncated_normal([3, 2, 2, 16, 16], stddev=0.1))
    W_conv2_7_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 16], stddev=0.1))
    W_conv2_7_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 32], stddev=0.1))

    b_conv2_7_1 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_7_2 = tf.Variable(tf.constant(0.1, shape=[16]))
    b_conv2_7_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 3rd strategy 1st block 20 54 46 32
    W_conv3_1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_1_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_1_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))

    b_conv3_1_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_1_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_1_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # # 3rd strategy 2nd block 20 54 46 32
    # W_conv3_2_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    # W_conv3_2_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    # W_conv3_2_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    #
    # b_conv3_2_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    # b_conv3_2_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    # b_conv3_2_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # # 3rd strategy 3rd block 20 54 46 32
    # W_conv3_3_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    # W_conv3_3_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    # W_conv3_3_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    #
    # b_conv3_3_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    # b_conv3_3_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    # b_conv3_3_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 3rd strategy 4th block 20 54 46 32
    W_conv3_4_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_4_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_4_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))

    b_conv3_4_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_4_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_4_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 3rd strategy 5th block 20 54 46 32
    W_conv3_5_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_5_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_5_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))

    b_conv3_5_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_5_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_5_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 3rd strategy 6th block 20 54 46 32
    W_conv3_6_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_6_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_6_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))

    b_conv3_6_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_6_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_6_3 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 3rd strategy 7th block 20 54 46 32 -> 10 27 23 64
    W_conv3_7_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_7_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 32], stddev=0.1))
    W_conv3_7_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 32, 64], stddev=0.1))

    b_conv3_7_1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_7_2 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv3_7_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 4th strategy 1st block 10 27 23 64 -> 10 27 23 64
    W_conv4_1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_1_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_1_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))

    b_conv4_1_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_1_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_1_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # # 4th strategy 2nd block 10 27 23 64 -> 10 27 23 64
    # W_conv4_2_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    # W_conv4_2_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    # W_conv4_2_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    #
    # b_conv4_2_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    # b_conv4_2_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    # b_conv4_2_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # # 4th strategy 3rd block 10 27 23 64 -> 10 27 23 64
    # W_conv4_3_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    # W_conv4_3_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    # W_conv4_3_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    #
    # b_conv4_3_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    # b_conv4_3_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    # b_conv4_3_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 4th strategy 4th block 10 27 23 64 -> 10 27 23 64
    W_conv4_4_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_4_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_4_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))

    b_conv4_4_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_4_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_4_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 4th strategy 5th block 10 27 23 64 -> 10 27 23 64
    W_conv4_5_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_5_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_5_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))

    b_conv4_5_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_5_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_5_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 4th strategy 6th block 10 27 23 64
    W_conv4_6_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_6_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_6_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))

    b_conv4_6_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_6_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_6_3 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 4th strategy 7th block 10 27 23 64 -> 4 13 11 128
    W_conv4_7_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_7_2 = tf.Variable(tf.truncated_normal([3, 2, 2, 64, 64], stddev=0.1))
    W_conv4_7_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1))
    W_conv4_7_4 = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 128], stddev=0.1))

    b_conv4_7_1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_7_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_7_3 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv4_7_4 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 5th strategy 1st block 4 13 11 128 -> 4 13 11 128
    W_conv5_1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_1_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_1_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))

    b_conv5_1_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_1_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_1_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # # 5th strategy 2nd block 4 13 11 128 -> 4 13 11 128
    # W_conv5_2_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    # W_conv5_2_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    # W_conv5_2_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    #
    # b_conv5_2_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    # b_conv5_2_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    # b_conv5_2_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # # 5th strategy 3rd block 4 13 11 128 -> 4 13 11 128
    # W_conv5_3_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    # W_conv5_3_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    # W_conv5_3_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    #
    # b_conv5_3_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    # b_conv5_3_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    # b_conv5_3_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 5th strategy 4th block 4 13 11 128 -> 4 13 11 128
    W_conv5_4_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_4_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_4_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))

    b_conv5_4_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_4_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_4_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 5th strategy 5th block 4 13 11 128 -> 4 13 11 128
    W_conv5_5_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_5_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_5_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))

    b_conv5_5_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_5_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_5_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 5th strategy 6th block 4 13 11 128
    W_conv5_6_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_6_2 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_6_3 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))

    b_conv5_6_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_6_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_6_3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 5th strategy 7th block 4 13 11 128 -> 1 6 5 256
    W_conv5_7_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1))
    W_conv5_7_2 = tf.Variable(tf.truncated_normal([4, 2, 2, 128, 128], stddev=0.1))
    W_conv5_7_3 = tf.Variable(tf.truncated_normal([1, 3, 3, 128, 128], stddev=0.1))
    W_conv5_7_4 = tf.Variable(tf.truncated_normal([1, 3, 3, 128, 256], stddev=0.1))

    b_conv5_7_1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_7_2 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_7_3 = tf.Variable(tf.constant(0.1, shape=[128]))
    b_conv5_7_4 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 6th strategy 1st block 1 6 5 256 -> 1 6 5 256
    W_conv6_1_1 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    W_conv6_1_2 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    W_conv6_1_3 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))

    b_conv6_1_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv6_1_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv6_1_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # # 6th strategy 2nd block 1 6 5 256
    # W_conv6_2_1 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    # W_conv6_2_2 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    # W_conv6_2_3 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    #
    # b_conv6_2_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    # b_conv6_2_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    # b_conv6_2_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 6th strategy 3rd block 1 6 5 256 -> 1 2 2 512
    W_conv6_3_1 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    W_conv6_3_2 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    W_conv6_3_3 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))

    b_conv6_3_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv6_3_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv6_3_3 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 6th strategy 4th block 1 6 5 256 -> 1 2 2 512
    W_conv6_4_1 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    W_conv6_4_2 = tf.Variable(tf.truncated_normal([1, 3, 2, 256, 256], stddev=0.1))
    W_conv6_4_3 = tf.Variable(tf.truncated_normal([1, 3, 3, 256, 256], stddev=0.1))
    W_conv6_4_4 = tf.Variable(tf.truncated_normal([1, 2, 2, 256, 512], stddev=0.1))

    b_conv6_4_1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv6_4_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv6_4_3 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_conv6_4_4 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 7th strategy 1st block 1 2 2 512 -> 1 1 1 512
    W_conv7_1 = tf.Variable(tf.truncated_normal([1, 2, 2, 512, 512], stddev=0.1))
    W_conv7_2 = tf.Variable(tf.truncated_normal([1, 2, 2, 512, 512], stddev=0.1))
    W_conv7_3 = tf.Variable(tf.truncated_normal([1, 1, 1, 512, 512], stddev=0.1))

    b_conv7_1 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv7_2 = tf.Variable(tf.constant(0.1, shape=[512]))
    b_conv7_3 = tf.Variable(tf.constant(0.1, shape=[512]))

    x_CNN = tf.reshape(x, shape=[-1, 84, 218, 146, 1])

    # 42 109 73 8
    conv_1_output = get_Convolution_1(x_CNN, W_conv1_1, b_conv1_1)

    conv_2_1_input = conv_1_output

    # 42 109 73 16
    conv_2_1_output = get_Convolution_2(conv_2_1_input, W_conv2_1_1, b_conv2_1_1, W_conv2_1_2, b_conv2_1_2, W_conv2_1_3,
                                        b_conv2_1_3, block=0)

    conv_2_1_output = get_Fusion_pool_add_channel(conv_2_1_input, conv_2_1_output, 0, 0, pad=0, add_channel=1)

    conv_2_4_input = conv_2_1_output

    # # 42 109 73 16
    # conv_2_2_output = get_Convolution_2(conv_2_2_input, W_conv2_2_1, b_conv2_2_1, W_conv2_2_2, b_conv2_2_2, W_conv2_2_3,
    #                                     b_conv2_2_3, block=0)
    #
    # conv_2_2_output = get_Fusion_pool_add_channel(conv_2_2_input, conv_2_2_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_2_3_input = conv_2_2_output

    # # 42 109 73 16
    # conv_2_3_output = get_Convolution_2(conv_2_3_input, W_conv2_3_1, b_conv2_3_1, W_conv2_3_2, b_conv2_3_2, W_conv2_3_3,
    #                                     b_conv2_3_3, block=0)
    #
    # conv_2_3_output = get_Fusion_pool_add_channel(conv_2_3_input, conv_2_3_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_2_4_input = conv_2_3_output

    # 42 109 73 16
    conv_2_4_output = get_Convolution_2(conv_2_4_input, W_conv2_4_1, b_conv2_4_1, W_conv2_4_2, b_conv2_4_2, W_conv2_4_3,
                                        b_conv2_4_3, block=0)

    conv_2_4_output = get_Fusion_pool_add_channel(conv_2_4_input, conv_2_4_output, 0, 0, pad=0, add_channel=0)

    conv_2_5_input = conv_2_4_output

    # 42 109 73 16
    conv_2_5_output = get_Convolution_2(conv_2_5_input, W_conv2_5_1, b_conv2_5_1, W_conv2_5_2, b_conv2_5_2, W_conv2_5_3,
                                        b_conv2_5_3, block=0)

    conv_2_5_output = get_Fusion_pool_add_channel(conv_2_5_input, conv_2_5_output, 0, 0, pad=0, add_channel=0)

    conv_2_6_input = conv_2_5_output

    # 42 109 73 16
    conv_2_6_output = get_Convolution_2(conv_2_6_input, W_conv2_6_1, b_conv2_6_1, W_conv2_6_2, b_conv2_6_2, W_conv2_6_3,
                                        b_conv2_6_3, block=0)

    conv_2_6_output = get_Fusion_pool_add_channel(conv_2_6_input, conv_2_6_output, 0, 0, pad=0, add_channel=0)

    conv_2_7_input = conv_2_6_output

    # 20 54 46 32
    conv_2_7_output = get_Convolution_2(conv_2_7_input, W_conv2_7_1, b_conv2_7_1, W_conv2_7_2, b_conv2_7_2, W_conv2_7_3,
                                        b_conv2_7_3, block=1)

    conv_2_7_output = get_Fusion_pool_add_channel(conv_2_7_input, conv_2_7_output, [1, 2, 2, 2, 1], [1, 3, 2, 2, 1],
                                                  pad=2, add_channel=1)

    conv_3_1_input = conv_2_7_output

    # 20 54 46 32
    conv_3_1_output = get_Convolution_3(conv_3_1_input, W_conv3_1_1, b_conv3_1_1, W_conv3_1_2, b_conv3_1_2, W_conv3_1_3,
                                        b_conv3_1_3)

    conv_3_1_output = get_Fusion_pool_add_channel(conv_3_1_input, conv_3_1_output, 0, 0, pad=0, add_channel=0)

    conv_3_4_input = conv_3_1_output

    # # 20 54 46 32
    # conv_3_2_output = get_Convolution_3(conv_3_2_input, W_conv3_2_1, b_conv3_2_1, W_conv3_2_2, b_conv3_2_2, W_conv3_2_3,
    #                                     b_conv3_2_3)
    #
    # conv_3_2_output = get_Fusion_pool_add_channel(conv_3_2_input, conv_3_2_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_3_3_input = conv_3_2_output

    # # 20 54 46 32
    # conv_3_3_output = get_Convolution_3(conv_3_3_input, W_conv3_3_1, b_conv3_3_1, W_conv3_3_2, b_conv3_3_2, W_conv3_3_3,
    #                                     b_conv3_3_3)
    #
    # conv_3_3_output = get_Fusion_pool_add_channel(conv_3_3_input, conv_3_3_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_3_4_input = conv_3_3_output

    # 20 54 46 32
    conv_3_4_output = get_Convolution_3(conv_3_4_input, W_conv3_4_1, b_conv3_4_1, W_conv3_4_2, b_conv3_4_2,
                                        W_conv3_4_3, b_conv3_4_3)

    conv_3_4_output = get_Fusion_pool_add_channel(conv_3_4_input, conv_3_4_output, 0, 0, pad=0, add_channel=0)

    conv_3_5_input = conv_3_4_output

    # 20 54 46 32
    conv_3_5_output = get_Convolution_3(conv_3_5_input, W_conv3_5_1, b_conv3_5_1, W_conv3_5_2, b_conv3_5_2,
                                        W_conv3_5_3, b_conv3_5_3)

    conv_3_5_output = get_Fusion_pool_add_channel(conv_3_5_input, conv_3_5_output, 0, 0, pad=0, add_channel=0)

    conv_3_6_input = conv_3_5_output

    # 20 54 46 32
    conv_3_6_output = get_Convolution_3(conv_3_6_input, W_conv3_6_1, b_conv3_6_1, W_conv3_6_2, b_conv3_6_2,
                                        W_conv3_6_3, b_conv3_6_3)

    conv_3_6_output = get_Fusion_pool_add_channel(conv_3_6_input, conv_3_6_output, 0, 0, pad=0, add_channel=0)

    conv_3_7_input = conv_3_6_output

    # 10 27 23 64
    conv_3_7_output = get_Convolution_3_pool(conv_3_7_input, W_conv3_7_1, b_conv3_7_1, W_conv3_7_2, b_conv3_7_2,
                                             W_conv3_7_3, b_conv3_7_3)

    conv_3_7_output = get_Fusion_pool_add_channel(conv_3_7_input, conv_3_7_output, [1, 2, 2, 2, 1], 0, pad=1,
                                                  add_channel=1)

    conv_4_1_input = conv_3_7_output

    # 10 27 23 64
    conv_4_1_output = get_Convolution_X(conv_4_1_input, W_conv4_1_1, b_conv4_1_1, W_conv4_1_2, b_conv4_1_2, W_conv4_1_3,
                                        b_conv4_1_3)

    conv_4_1_output = get_Fusion_pool_add_channel(conv_4_1_input, conv_4_1_output, 0, 0, pad=0, add_channel=0)

    conv_4_4_input = conv_4_1_output

    # # 10 27 23 64
    # conv_4_2_output = get_Convolution_X(conv_4_2_input, W_conv4_2_1, b_conv4_2_1, W_conv4_2_2, b_conv4_2_2, W_conv4_2_3,
    #                                     b_conv4_2_3)
    #
    # conv_4_2_output = get_Fusion_pool_add_channel(conv_4_2_input, conv_4_2_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_4_3_input = conv_4_2_output

    # # 10 27 23 64
    # conv_4_3_output = get_Convolution_X(conv_4_3_input, W_conv4_3_1, b_conv4_3_1, W_conv4_3_2, b_conv4_3_2, W_conv4_3_3,
    #                                     b_conv4_3_3)
    #
    # conv_4_3_output = get_Fusion_pool_add_channel(conv_4_3_input, conv_4_3_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_4_4_input = conv_4_3_output

    # 10 27 23 64
    conv_4_4_output = get_Convolution_X(conv_4_4_input, W_conv4_4_1, b_conv4_4_1, W_conv4_4_2, b_conv4_4_2, W_conv4_4_3,
                                        b_conv4_4_3)

    conv_4_4_output = get_Fusion_pool_add_channel(conv_4_4_input, conv_4_4_output, 0, 0, pad=0, add_channel=0)

    conv_4_5_input = conv_4_4_output

    # 10 27 23 64
    conv_4_5_output = get_Convolution_X(conv_4_5_input, W_conv4_5_1, b_conv4_5_1, W_conv4_5_2, b_conv4_5_2, W_conv4_5_3,
                                        b_conv4_5_3)

    conv_4_5_output = get_Fusion_pool_add_channel(conv_4_5_input, conv_4_5_output, 0, 0, pad=0, add_channel=0)

    conv_4_6_input = conv_4_5_output

    # 10 27 23 64
    conv_4_6_output = get_Convolution_X(conv_4_6_input, W_conv4_6_1, b_conv4_6_1, W_conv4_6_2, b_conv4_6_2, W_conv4_6_3,
                                        b_conv4_6_3)

    conv_4_6_output = get_Fusion_pool_add_channel(conv_4_6_input, conv_4_6_output, 0, 0, pad=0, add_channel=0)

    conv_4_7_input = conv_4_6_output

    # 4 13 11 128
    conv_4_7_output = get_Convolution_X_pool(conv_4_7_input, W_conv4_7_1, b_conv4_7_1, W_conv4_7_2, b_conv4_7_2,
                                             W_conv4_7_3, b_conv4_7_3, W_conv4_7_4, b_conv4_7_4)

    conv_4_7_output = get_Fusion_pool_add_channel(conv_4_7_input, conv_4_7_output, [1, 2, 2, 2, 1], [1, 3, 2, 2, 1],
                                                  pad=2, add_channel=1)

    conv_5_1_input = conv_4_7_output

    # 4 13 11 128
    conv_5_1_output = get_Convolution_X(conv_5_1_input, W_conv5_1_1, b_conv5_1_1, W_conv5_1_2, b_conv5_1_2, W_conv5_1_3,
                                        b_conv5_1_3)

    conv_5_1_output = get_Fusion_pool_add_channel(conv_5_1_input, conv_5_1_output, 0, 0, pad=0, add_channel=0)

    conv_5_4_input = conv_5_1_output

    # # 4 13 11 128
    # conv_5_2_output = get_Convolution_X(conv_5_2_input, W_conv5_2_1, b_conv5_2_1, W_conv5_2_2, b_conv5_2_2, W_conv5_2_3,
    #                                     b_conv5_2_3)
    #
    # conv_5_2_output = get_Fusion_pool_add_channel(conv_5_2_input, conv_5_2_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_5_3_input = conv_5_2_output

    # # 4 13 11 128
    # conv_5_3_output = get_Convolution_X(conv_5_3_input, W_conv5_3_1, b_conv5_3_1, W_conv5_3_2, b_conv5_3_2, W_conv5_3_3,
    #                                     b_conv5_3_3)
    #
    # conv_5_3_output = get_Fusion_pool_add_channel(conv_5_3_input, conv_5_3_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_5_4_input = conv_5_3_output

    # 4 13 11 128
    conv_5_4_output = get_Convolution_X(conv_5_4_input, W_conv5_4_1, b_conv5_4_1, W_conv5_4_2, b_conv5_4_2, W_conv5_4_3,
                                        b_conv5_4_3)

    conv_5_4_output = get_Fusion_pool_add_channel(conv_5_4_input, conv_5_4_output, 0, 0, pad=0, add_channel=0)

    conv_5_5_input = conv_5_4_output

    # 4 13 11 128
    conv_5_5_output = get_Convolution_X(conv_5_5_input, W_conv5_5_1, b_conv5_5_1, W_conv5_5_2, b_conv5_5_2, W_conv5_5_3,
                                        b_conv5_5_3)

    conv_5_5_output = get_Fusion_pool_add_channel(conv_5_5_input, conv_5_5_output, 0, 0, pad=0, add_channel=0)

    conv_5_6_input = conv_5_5_output

    # 4 13 11 128
    conv_5_6_output = get_Convolution_X(conv_5_6_input, W_conv5_6_1, b_conv5_6_1, W_conv5_6_2, b_conv5_6_2, W_conv5_6_3,
                                        b_conv5_6_3)

    conv_5_6_output = get_Fusion_pool_add_channel(conv_5_6_input, conv_5_6_output, 0, 0, pad=0, add_channel=0)

    conv_5_7_input = conv_5_6_output

    # 1 6 5 256
    conv_5_7_output = get_Convolution_X_pool(conv_5_7_input, W_conv5_7_1, b_conv5_7_1, W_conv5_7_2, b_conv5_7_2,
                                             W_conv5_7_3, b_conv5_7_3, W_conv5_7_4, b_conv5_7_4)

    conv_5_7_output = get_Fusion_pool_add_channel(conv_5_7_input, conv_5_7_output, [1, 2, 2, 2, 1], [1, 4, 2, 2, 1],
                                                  pad=2, add_channel=1)

    conv_6_1_input = conv_5_7_output

    # 1 6 5 256
    conv_6_1_output = get_Convolution_X(conv_6_1_input, W_conv6_1_1, b_conv6_1_1, W_conv6_1_2, b_conv6_1_2, W_conv6_1_3,
                                        b_conv6_1_3)

    conv_6_1_output = get_Fusion_pool_add_channel(conv_6_1_input, conv_6_1_output, 0, 0, pad=0, add_channel=0)

    conv_6_3_input = conv_6_1_output

    # # 1 6 5 256
    # conv_6_2_output = get_Convolution_X(conv_6_2_input, W_conv6_2_1, b_conv6_2_1, W_conv6_2_2, b_conv6_2_2, W_conv6_2_3,
    #                                     b_conv6_2_3)
    #
    # conv_6_2_output = get_Fusion_pool_add_channel(conv_6_2_input, conv_6_2_output, 0, 0, pad=0, add_channel=0)
    #
    # conv_6_3_input = conv_6_2_output

    # 1 6 5 256
    conv_6_3_output = get_Convolution_X(conv_6_3_input, W_conv6_3_1, b_conv6_3_1, W_conv6_3_2, b_conv6_3_2, W_conv6_3_3,
                                        b_conv6_3_3)

    conv_6_3_output = get_Fusion_pool_add_channel(conv_6_3_input, conv_6_3_output, 0, 0, pad=0, add_channel=0)

    conv_6_4_input = conv_6_3_output

    # 1 2 2 512
    conv_6_4_output = get_Convolution_X_pool(conv_6_4_input, W_conv6_4_1, b_conv6_4_1, W_conv6_4_2, b_conv6_4_2,
                                             W_conv6_4_3, b_conv6_4_3, W_conv6_4_4, b_conv6_4_4)

    conv_6_4_output = get_Fusion_pool_add_channel(conv_6_4_input, conv_6_4_output, [1, 2, 2, 2, 1], [1, 1, 3, 2, 1],
                                                  pad=2, add_channel=1)

    conv_7_input = conv_6_4_output

    # 1 1 1 512
    conv_7_output = get_Convolution_X_last_layer(conv_7_input, W_conv7_1, b_conv7_1, W_conv7_2, b_conv7_2, W_conv7_3, b_conv7_3)

    W_fc = tf.Variable(tf.truncated_normal([1 * 1 * 512, 1024], stddev=0.1))
    W_linear = tf.Variable(tf.random_normal([1024, 2], stddev=0.1))

    b_fc = tf.Variable(tf.constant(0.1, shape=[1024]))
    b_linear = tf.Variable(tf.constant(0.1, shape=[2]))

    conv_7_vect = tf.reshape(conv_7_output, [-1, 1 * 1 * 512])

    non_fc = get_NonLinearLayer(get_LinearLayer(conv_7_vect, W_fc, b_fc))

    non_fc = tf.nn.dropout(non_fc, keep_prob)

    L_out = get_LinearLayer(non_fc, W_linear, b_linear)

    return L_out


def save_model(session, ith_net):
    file_name = './3D_model_2class_' + str(int(ith_net)) + '/'
    document_name = file_name + 'model.checkpoint'
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    saver = tf.train.Saver()
    saver.save(session, document_name)


def main():

    X = tf.placeholder("float", [None, 84, 218, 146])
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

                batch_x = np.zeros((6, 84, 218, 146))

                label_1_id_train_new = random.sample(range(535), 3)
                label_3_id_train_new = random.sample(range(212), 3)

                for k in range(3):
                    batch_x[k] = label_1_data[label_1_id_train_new[k]]
                    batch_x[k + 3] = label_3_data[label_3_id_train_new[k]]

                _, each_loss = sess.run([optimize, loss], feed_dict={X: batch_x, Y: label_for_train_and_test, LEARNING_RATE: LR})
                total_loss[indexIter] = each_loss
                print('ith_net', ith_net)
                print('indexIter %d: Loss %.5f' % (indexIter, each_loss))

                if (indexIter+1) % train_batch == 0:

                    index_acc = (indexIter + 1) / train_batch

                    accuracy_test = 0
                    for test_indexIter in range(23):

                        testX = np.zeros((6, 84, 218, 146))

                        lower_index_test = int(test_indexIter * 3)
                        upper_index_test = int((test_indexIter + 1) * 3)

                        label_1_id_test = label_1_id_test_new[lower_index_test:upper_index_test]
                        label_3_id_test = label_3_id_test_new[lower_index_test:upper_index_test]

                        for k in range(3):
                            testX[k] = label_1_data[label_1_id_test[k]]
                            testX[k + 3] = label_3_data[label_3_id_test[k]]

                        accuracy_test_t, pp = sess.run([accuracy, predict], feed_dict={X: testX, Y: label_for_train_and_test})
                        ppv = np.argmax(pp, 1)
                        print(test_indexIter)
                        print(ppv)
                        accuracy_test += accuracy_test_t

                    accuracy_test /= test_batch
                    error_test = 1 - accuracy_test

                    print('Iteration %d: Accuracy %.5f Error %.5f (test)' % (indexIter, accuracy_test, error_test))

                    test_accuracy[int(index_acc-1)] = accuracy_test

                if indexIter >= 3550:
                    LR = 0.001 * (1 - (indexIter + 1 - 3550)/iter_num)

            save_model(sess, ith_net)
            document_name = '3D_2class_curve_' + str(int(ith_net)) + '.npz'
            np.savez(document_name, total_loss=total_loss, test_accuracy=test_accuracy)

if __name__ == '__main__':
    main()
