# -*- coding: utf-8 -*-
import tensorflow as tf

epsilon = 1e-9


def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def fc_capsule(sep_encodes, mask):
    attn = softmax(tf.layers.dense(sep_encodes, 1, activation=tf.nn.tanh) + mask, 1)
    return tf.reduce_sum(attn * sep_encodes, 1)


def classify_capsule(fuse_encodes):
    attn = tf.expand_dims(softmax(tf.layers.dense(fuse_encodes, 3, activation=tf.nn.tanh), 1), 2)
    return tf.reduce_sum(attn * tf.expand_dims(fuse_encodes, 3), 1)


def concat_capsules(capsule_af, capsule_as, capsule_at, mask_f, mask_s, mask_t):
    # batch , dim  ,3
    concat_capsule = tf.concat(
        [tf.expand_dims(t, 2) for t in
         [fc_capsule(capsule_af, mask_f), fc_capsule(capsule_as, mask_s), fc_capsule(capsule_at, mask_t)]], 2)
    # batch ,1 , dim , 1 ,3
    expand_capsules = tf.expand_dims(concat_capsule, 1)
    alter_capsules = tf.expand_dims(expand_capsules, 3)
    return alter_capsules


def routing(conv_capsules, alter_capsule, dim_size, conv_nums):
    inputs = tf.expand_dims(conv_capsules, 4)
    # w [batch size, time, dim ,channels, out capsules]
    caps_w = tf.get_variable('caps_weight', shape=(1, 1, dim_size, conv_nums, 3), dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=0.1))
    caps_b = tf.get_variable('caps_bias', shape=(1, 1, dim_size, conv_nums, 3))
    # u [batch size, time, dim ,channels, out capsules]
    u_hat = caps_w * inputs + caps_b
    u_hat = u_hat * alter_capsule
    # batch , time , 1 , channels , out capsules
    b_ij = reduce_sum(0 * u_hat, axis=2, keepdims=True)
    iter_time = 3
    for r_iter in range(iter_time):
        with tf.variable_scope('iter_' + str(r_iter)):
            # softmax
            c_ij = softmax(b_ij, axis=4)
            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_time - 1:
                s_j = tf.multiply(c_ij, u_hat)
                s_j = reduce_sum(s_j, axis=3)
                s_j = reduce_sum(s_j, axis=1)
                v_j = squash(s_j)
            else:  # Inner iterations, do not apply backpropagation
                s_j = tf.multiply(c_ij, u_hat)
                s_j = reduce_sum(s_j, axis=1, keepdims=True)
                s_j = reduce_sum(s_j, axis=3, keepdims=True)
                v_j = squash(s_j)

                u_produce_v = reduce_sum(v_j * u_hat, axis=2, keepdims=True)

                # b_ij += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_ij += u_produce_v

    return v_j


def squash(vector):
    vec_squared_norm = reduce_sum(tf.square(vector), -3, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed
