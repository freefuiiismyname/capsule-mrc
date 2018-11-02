# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as tc


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True, state=None,
        history=False):
    """
    实现 (Bi-)LSTM, (Bi-)GRU 和 (Bi-)RNN
    """
    if history:
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c, h = state
                state = h
        else:
            # 双向lstm,前向细胞、反向细胞
            cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32, initial_state_fw=state,
                initial_state_bw=state
             )
        # 获取双向状态
            state_fw, state_bw = state
            if rnn_type.endswith('lstm'):
                c_fw, h_fw = state_fw
                c_bw, h_bw = state_bw
                # 双向历史信息
                state_fw, state_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                state = tf.concat([state_fw, state_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                state = state_fw + state_bw
        return outputs, state
    else:
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c, h = state
                state = h
        else:
            # 双向lstm,前向细胞、反向细胞
            cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
             )
        # 获取双向状态
            state_fw, state_bw = state
            state = state_fw
            if concat:
                outputs = tf.concat(outputs, 2)
            else:
                outputs = outputs[0] + outputs[1]
        return outputs, state


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    获取循环神经网络的细胞
    """
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    if layer_num > 1:
        cell = tc.rnn.MultiRNNCell([cell]*layer_num, state_is_tuple=True)
    return cell


