# -*- coding:utf8 -*-

import os
import time
import logging
import json
import tensorflow as tf
from layers.basic_rnn import rnn
from layers.caps_layer import routing
from layers.caps_layer import softmax
from layers.caps_layer import concat_capsules
from layers.match_layer import AttentionFlowMatchLayer
import keras.backend as K
import numpy as np
import random
import shutil


def cast(odtensor):
    return tf.expand_dims(tf.cast(odtensor, tf.float32), 1)


class RCModel(object):
    """
    实现阅读理解模型
    """

    def __init__(self, vocab, args):

        # 日志
        self.logger = logging.getLogger("brc")

        # 基础设置
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # 长度限制
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # 词汇表
        self.vocab = vocab

        # 会话信息
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # 存储信息
        self.saver = tf.train.Saver()

        # 模型初始化
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        建立tf的运算图
        1.初始化占位符 2.词嵌入 3.编码 4.匹配 5.融合 6.解码 7.计算损失 8.创建训练节点
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._conv()
        self._capsule()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('建立运算图用时为: {}  秒'.format(time.time() - start_t))

    def _setup_placeholders(self):
        """
        占位符
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.a_f = tf.placeholder(tf.int32, [None, None])
        self.a_s = tf.placeholder(tf.int32, [None, None])
        self.a_t = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.label_answer = tf.placeholder(tf.int32, [None])
        self.choose_type = tf.placeholder(tf.float32, [None])
        self.a_f_length = tf.placeholder(tf.int32, [None])
        self.a_s_length = tf.placeholder(tf.int32, [None])
        self.a_t_length = tf.placeholder(tf.int32, [None])

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

    def _embed(self):
        """
        嵌入层.问题和文章共享同样的嵌入方式
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.pre_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size() - 1, self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings[1:]),
                trainable=False
            )
            padding_vec = tf.Variable(tf.random_uniform([1, 300], -0.05, 0.05))
            # 只对填充向量进行训练，其余向量保持word2vec的结果
            self.word_embeddings = tf.concat([padding_vec, self.pre_embeddings], 0)
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
            self.a_f_emb = tf.nn.embedding_lookup(self.word_embeddings, self.a_f)
            self.a_s_emb = tf.nn.embedding_lookup(self.word_embeddings, self.a_s)
            self.a_t_emb = tf.nn.embedding_lookup(self.word_embeddings, self.a_t)

            self.q_mask = tf.expand_dims(tf.cast(tf.logical_not(tf.cast(self.q, tf.bool)), tf.float32) * -9999, -1)
            self.p_mask = tf.expand_dims(tf.cast(tf.logical_not(tf.cast(self.p, tf.bool)), tf.float32) * -9999, -1)
            self.a_f_mask = tf.expand_dims(tf.cast(tf.logical_not(tf.cast(self.a_f, tf.bool)), tf.float32) * -9999, -1)
            self.a_s_mask = tf.expand_dims(tf.cast(tf.logical_not(tf.cast(self.a_s, tf.bool)), tf.float32) * -9999, -1)
            self.a_t_mask = tf.expand_dims(tf.cast(tf.logical_not(tf.cast(self.a_t, tf.bool)), tf.float32) * -9999, -1)

    def _encode(self):
        """
        使用几个双向LSTM分别对问题、文章和候选答案编码
        问题作为历史信息要流入到文章、候选答案中
        """

        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, self.question_state = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size,
                                        state=self.question_state, history=True)
        with tf.variable_scope('answer_f_encoding'):
            self.sep_af_encodes, _ = rnn('bi-lstm', self.a_f_emb, self.a_f_length, self.hidden_size,
                                         state=self.question_state, history=True)
        with tf.variable_scope('answer_s_encoding'):
            self.sep_as_encodes, _ = rnn('bi-lstm', self.a_s_emb, self.a_s_length, self.hidden_size,
                                         state=self.question_state, history=True)
        with tf.variable_scope('answer_t_encoding'):
            self.sep_at_encodes, _ = rnn('bi-lstm', self.a_t_emb, self.a_t_length, self.hidden_size,
                                         state=self.question_state, history=True)

    def _match(self):
        """
        使用multiway attn 和 bidaf 来交互问题与文章
        """
        # batch size, q time , p time
        match_layer = AttentionFlowMatchLayer(self.hidden_size)
        match_p_encodes, self.tq_emb = match_layer.match(self.sep_p_encodes, self.sep_q_encodes)
        self.tp_emb = tf.concat([self.p_emb, match_p_encodes], axis=-1)

    def _fuse(self):
        """
        match之后，使用双向lstm来融合上下文信息
        """
        # 聚类信息fuse
        with tf.variable_scope('p-routing-fusion'):
            self.routing_p_encodes, _ = rnn('bi-lstm', self.tp_emb, self.p_length, self.hidden_size)

    def _conv(self):
        """
        利用卷积融合句意，每个卷积核代表一种独立看待问题的角度
        """
        # 分类信息提取capsule
        pooled_outputs = []
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with tf.variable_scope('p-conv'):
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.hidden_size * 2, 1, 300]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[300]), name="b")
                    conv = tf.nn.conv2d(tf.expand_dims(self.routing_p_encodes, -1), W, strides=[1, 1, 1, 1],
                                        padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.reduce_max(h, 1)
                    pooled_outputs.append(tf.expand_dims(pooled, -1))
        self.h_pool = tf.concat(pooled_outputs, 3)

    def _capsule(self):
        # 动态路由
        with tf.variable_scope('dynamic-routing'):
            alt_cap = concat_capsules(self.sep_af_encodes, self.sep_as_encodes, self.sep_at_encodes, self.a_f_mask,
                                      self.a_s_mask, self.a_t_mask)
            cls_cap = tf.Variable(tf.truncated_normal([1, 1, self.hidden_size * 2, 1, 3]))
            cap_gate = tf.reshape(tf.layers.dense(tf.concat(
                [tf.reshape(alt_cap, [-1, 6 * self.hidden_size]), tf.expand_dims(self.choose_type, 1),
                 cast(self.p_length), cast(self.q_length)], -1), 1, activation=tf.nn.sigmoid), [-1, 1, 1, 1, 1])
            self.ref_cap = alt_cap * cap_gate + cls_cap * (1 - cap_gate)
            # 聚类
            with tf.variable_scope('r-routing'):
                self.ar_r_capsule = routing(self.h_pool, self.ref_cap, self.hidden_size * 2, 10)

    def _decode(self):
        """
        解码顶层capsules，根据其模长softmax作为答案的概率
        """
        # 根据下层capsule构建候选答案capsule
        with tf.variable_scope('concat-capsule'):
            fc_weights = tf.Variable(tf.truncated_normal([1, 300, 3, 3]))
            fc_bias = tf.Variable(tf.truncated_normal([1, 300, 3]))
            fc_capsules = tf.expand_dims(self.ar_r_capsule, 3) * fc_weights
            self.concat_alter_capsule = tf.reduce_sum(fc_capsules, 2) + fc_bias
        # 候选答案间信息交互，用以缩放
        with tf.variable_scope('gated-capsule'):
            fc_gate = tf.reshape(tf.layers.dense(
                tf.concat([tf.reshape(self.concat_alter_capsule, [-1, 900]), tf.expand_dims(self.choose_type, 1),
                                      cast(self.p_length), cast(self.q_length)], -1), 3, activation=tf.nn.sigmoid),
                                 [-1, 1, 3])
            self.final_alter_capsule = self.concat_alter_capsule * fc_gate

            capsule_lengths = tf.sqrt(tf.reduce_sum(tf.square(self.final_alter_capsule), 1))
            self.alternatives_probs = softmax(capsule_lengths, 1)

    def _compute_loss(self):
        """
           计算损失
        """
        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            负对数似然损失
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.alter_loss = sparse_nll_loss(probs=self.alternatives_probs, labels=self.label_answer)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(self.alter_loss)
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        选择训练算法并以此创建一个训练操作
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('不支持的优化器: {}'.format(self.optim_type))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        单次训练模型
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.choose_type: batch['choose_type'],
                         self.q: batch['query_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['query_length'],
                         self.a_f_length: batch['alternatives_f_length'],
                         self.a_s_length: batch['alternatives_s_length'],
                         self.a_t_length: batch['alternatives_t_length'],
                         self.a_f: batch['alternative_f_token_ids'],
                         self.a_s: batch['alternative_s_token_ids'],
                         self.a_t: batch['alternative_t_token_ids'],
                         self.label_answer: batch['label_answer'],
                         self.dropout_keep_prob: dropout_keep_prob,
                         self.training: True,
                         }
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss

            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('第 {} 到第 {}批训练集的平均损失为 {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0):
        """
        训练模型
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        for epoch in range(1, epochs + 1):
            self.logger.info('第{}次训练模型 '.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('该批次的平均训练损失 {} is {}'.format(epoch, train_loss))
            eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
            self.evaluate_acc(eval_batches)
            self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate_acc(self, eval_batches):
        pred_answers = []
        true_answers = []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['query_token_ids'],
                         self.a_f: batch['alternative_f_token_ids'],
                         self.a_s: batch['alternative_s_token_ids'],
                         self.a_t: batch['alternative_t_token_ids'],
                         self.a_f_length: batch['alternatives_f_length'],
                         self.a_s_length: batch['alternatives_s_length'],
                         self.a_t_length: batch['alternatives_t_length'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['query_length'],
                         self.label_answer: batch['label_answer'],
                         self.choose_type: batch['choose_type'],
                         self.dropout_keep_prob: 1.0,
                         self.training: False
                         }

            alternatives_probs, loss = self.sess.run([
                self.alternatives_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            pred_answers.append(alternatives_probs)
            true_answers += batch['label_answer']

        pred_answers = np.concatenate(pred_answers, axis=0)
        pred_answers = np.argmax(pred_answers, axis=1)
        count = 0
        for i in range(len(pred_answers)):
            if pred_answers[i] == int(true_answers[i]):
                count += 1
        if len(pred_answers) > 0:
            acc = float(count) / float(len(true_answers))
        else:
            acc = 0
        self.logger.info('dev 集合acc {}'.format(acc))
        self.logger.info('dev 集合平均损失{}'.format(1.0 * total_loss / total_num))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None):
        """
        评估模型在验证集上的表现，如果指定了保存，则将把结果保存
        """
        pred_answers = []
        total_loss, total_num = 0, 0

        probs = []
        batches = []
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['query_token_ids'],
                         self.a_f: batch['alternative_f_token_ids'],
                         self.a_s: batch['alternative_s_token_ids'],
                         self.a_t: batch['alternative_t_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['query_length'],
                         self.a_f_length: batch['alternatives_f_length'],
                         self.a_s_length: batch['alternatives_s_length'],
                         self.a_t_length: batch['alternatives_t_length'],
                         self.label_answer: batch['label_answer'],
                         self.choose_type: batch['choose_type'],
                         self.dropout_keep_prob: 1.0,
                         }

            alternatives_probs, loss = self.sess.run([
                self.alternatives_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            batches += batch['raw_data']
            probs.append(alternatives_probs)
        probs = np.concatenate(probs, axis=0)

        for sample, prob in zip(batches, probs):
            best_answer_index = np.argmax(prob)
            pred_answer = ''
            for i in sample['segmented_alternatives'][best_answer_index]:
                pred_answer += i
            if 'answer' in sample:
                pred_answers.append({'query_id': sample['query_id'],
                                     'pred_answer': pred_answer,
                                     'query': sample['segmented_query'],
                                     'answer': sample['answer']}
                                    )
            else:
                pred_answers.append({'query_id': sample['query_id'],
                                     'pred_answer': pred_answer}
                                    )

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('保存 {} 结果到 {}'.format(result_prefix, result_file))

        # 该平均损失对于测试集是无效的，因为测试集没有标注答案
        ave_loss = 1.0 * total_loss / total_num

        return ave_loss,

    def save(self, model_dir, model_prefix):
        """
        保存模型
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('保存模型到 {}, 其前缀为 {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        重载模型
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('重载 {} 模型, 其前缀为 {}'.format(model_dir, model_prefix))
