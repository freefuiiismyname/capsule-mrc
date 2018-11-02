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


class SelfMatchingLayer(object):
    """
    Implements the self-matching layer.
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def getSelfMatchingCell(self, hidden_size, in_keep_prob=1.0):
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def match(self, passage_encodes, whole_passage_encodes, p_length):
        with tf.variable_scope('self-matching'):
            # 创建cell
            # whole_passage_encodes 作为整体匹配信息

            # cell_fw = SelfMatchingCell(self.hidden_size, question_encodes)
            # cell_bw = SelfMatchingCell(self.hidden_size, question_encodes)

            cell_fw = self.getSelfMatchingCell(self.hidden_size)
            cell_bw = self.getSelfMatchingCell(self.hidden_size)

            # function:

            # self.context_to_attend = whole_passage_encodes
            # fc_context = W * context_to_attend
            self.fc_context = tc.layers.fully_connected(whole_passage_encodes, num_outputs=self.hidden_size,
                                                        activation_fn=None)
            ref_vector = passage_encodes
            # 求St的tanh部分
            G = tf.tanh(self.fc_context + tf.expand_dims(
                tc.layers.fully_connected(ref_vector, num_outputs=self.hidden_size, activation_fn=None), 1))
            # tanh部分乘以bias
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            # 求a
            scores = tf.nn.softmax(logits, 1)
            # 求c
            attended_context = tf.reduce_sum(whole_passage_encodes * scores, axis=1)
            # birnn inputs
            input_encodes = tf.concat([ref_vector, attended_context], -1)
            """
            gated
            g_t = tf.sigmoid( tc.layers.fully_connected(whole_passage_encodes,num_outputs=self.hidden_size,activation_fn=None) )
            v_tP_c_t_star = tf.squeeze(tf.multiply(input_encodes , g_t))
            input_encodes = v_tP_c_t_star
            """

            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=input_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)

            match_outputs = tf.concat(outputs, 2)
            match_state = tf.concat([state, state], 1)

            # state_fw, state_bw = state
            # c_fw, h_fw = state_fw
            # c_bw, h_bw = state_bw
        return match_outputs, match_state
