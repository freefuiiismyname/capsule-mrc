# -*- coding: utf-8 -*-
"""
该模块实现Match-LSTM和BiDAF算法
"""

import tensorflow as tf
import tensorflow.contrib as tc
from .basic_rnn import rnn
import keras.backend as K


class AttentionFlowMatchLayer(object):
    """
    实现注意力流层来计算文本-问题、问题-文本的注意力
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.dim_size = hidden_size * 2

    """
        根据问题向量来匹配文章向量
    """

    def match(self, passage_encodes, question_encodes):
        with tf.variable_scope('attn-match'):
            # bidaf
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                            [1, tf.shape(passage_encodes)[1], 1])

            dnm_s1 = tf.expand_dims(passage_encodes, 2)
            dnm_s2 = tf.expand_dims(question_encodes, 1)

            # concat Attn
            sjt = tf.reduce_sum(dnm_s1 + dnm_s2, 3)
            ait = tf.nn.softmax(sjt, 2)
            qtc = tf.matmul(ait, question_encodes)

            # bi-linear Attn
            sjt = tf.matmul(passage_encodes, tf.transpose(question_encodes, perm=[0, 2, 1]))
            ait = tf.nn.softmax(sjt, 2)
            qtb = tf.matmul(ait, question_encodes)

            # dot Attn
            sjt = tf.reduce_sum(dnm_s1 * dnm_s2, 3)
            ait = tf.nn.softmax(sjt, 2)
            qtd = tf.matmul(ait, question_encodes)

            # minus Attn
            sjt = tf.reduce_sum(dnm_s1 - dnm_s2, 3)
            ait = tf.nn.softmax(sjt, 2)
            qtm = tf.matmul(ait, question_encodes)

            passage_outputs = tf.concat([passage_encodes, context2question_attn,
                                         passage_encodes * context2question_attn,
                                         passage_encodes * question2context_attn, qtc, qtb, qtd, qtm], -1)

        return passage_outputs, question_encodes
