# -*- coding:utf8 -*-

'''
This module implements data process strategies.
'''
import json
import logging
import numpy as np
import random

from nltk.translate.bleu_score import sentence_bleu


class BRCDataset(object):
    '''
    该模块实现加载和使用百度阅读理解数据集的api
    '''

    def __init__(self, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger('brc')
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file)
            self.logger.info('训练集规模: {} 个问题.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('验证集规模: {} 个问题.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('测试集规模: {} 个问题.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, sampling=False):
        '''
        加载数据集
        '''
        with open(data_path, encoding='utf-8') as fin:
            data_set = []
            filter_long_para, filter_long_query, filter_zero_query = 0, 0, 0
            for lidx, line in enumerate(fin):
                if sampling:
                    if random.randint(1, 10) > 1:
                        continue
                sample = json.loads(line.strip())
                if len(sample['segmented_passage']) > self.max_p_len:
                    filter_long_para += 1
                    continue
                if len(sample['segmented_query']) > self.max_q_len:
                    filter_long_query += 1
                    continue
                if len(sample['segmented_query']) == 0:
                    filter_zero_query += 1
                    continue
                if 'answer' in sample:
                    fake_label = sample['label_answer']
                    scores = []
                    alternatives = sample['alternatives'].split('|')
                    for alternative in alternatives:
                        score = 0
                        if '无法确定' in alternative or '无法确认' in alternative or '无法确的' in alternative:
                            score += 3
                        elif '不' in alternative or '没' in alternative:
                            score += 2
                        scores.append(score)
                    if sum(scores) < 5:
                        sample['choose_type'] = 1.0
                    else:
                        sample['choose_type'] = 0.0
                    f_index = scores.index(min(scores))
                    scores[f_index] = 10
                    s_index = scores.index(min(scores))
                    scores[s_index] = 10
                    t_index = scores.index(min(scores))
                    segmented_alternatives = [sample['segmented_alternatives'][f_index],
                                              sample['segmented_alternatives'][s_index],
                                              sample['segmented_alternatives'][t_index]]

                    sample['segmented_alternatives'] = segmented_alternatives
                    pos_alternatives = [sample['pos_alternatives'][f_index], sample['pos_alternatives'][s_index],
                                        sample['pos_alternatives'][t_index]]
                    sample['pos_alternatives'] = pos_alternatives
                    if f_index == fake_label:
                        sample['label_answer'] = 0
                    elif s_index == fake_label:
                        sample['label_answer'] = 1
                    else:
                        sample['label_answer'] = 2

                data_set.append(sample)

        print('passage超长过滤:', filter_long_para, 'query问题过滤:', filter_long_query + filter_zero_query)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        '''
        一个最小的批次
        '''
        batch_data = {'raw_data': [data[i] for i in indices],
                      'query_token_ids': [],
                      'query_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'alternative_f_token_ids': [],
                      'alternatives_f_length': [],
                      'alternative_s_token_ids': [],
                      'alternatives_s_length': [],
                      'alternative_t_token_ids': [],
                      'alternatives_t_length': [],
                      'alternative': [],
                      'label_answer': [],
                      'choose_type': []}
        # 将每个样本的信息都记录到batch里
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['query_token_ids'].append(sample['query_token_ids'])
            batch_data['query_length'].append(len(sample['query_token_ids']))
            batch_data['passage_token_ids'].append(sample['passage_token_ids'])
            batch_data['passage_length'].append(len(sample['passage_token_ids']))
            batch_data['alternative_f_token_ids'].append(sample['alternatives_token_ids'][0])
            batch_data['alternatives_f_length'].append(len(sample['alternatives_token_ids'][0]))
            batch_data['alternative_s_token_ids'].append(sample['alternatives_token_ids'][1])
            batch_data['alternatives_s_length'].append(len(sample['alternatives_token_ids'][1]))
            batch_data['alternative_t_token_ids'].append(sample['alternatives_token_ids'][2])
            batch_data['alternatives_t_length'].append(len(sample['alternatives_token_ids'][2]))
            batch_data['choose_type'].append(sample['choose_type'])
            try:
                batch_data['label_answer'].append(sample['label_answer'])
            except KeyError:
                batch_data['label_answer'].append(0)
        batch_data, padded_p_len, padded_q_len, padded_a_f_len, padded_a_s_len, padded_a_t_len = self._dynamic_padding(
            batch_data, pad_id)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        '''
        根据pad_id动态填充batch_data
        '''
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['query_length']))
        pad_a_f_len = max(batch_data['alternatives_f_length'])
        pad_a_s_len = max(batch_data['alternatives_s_length'])
        pad_a_t_len = max(batch_data['alternatives_t_length'])
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['query_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                         for ids in batch_data['query_token_ids']]
        batch_data['alternative_f_token_ids'] = [(ids + [pad_id] * (pad_a_f_len - len(ids)))[: pad_a_f_len]
                                                 for ids in batch_data['alternative_f_token_ids']]
        batch_data['alternative_s_token_ids'] = [(ids + [pad_id] * (pad_a_s_len - len(ids)))[: pad_a_s_len]
                                                 for ids in batch_data['alternative_s_token_ids']]
        batch_data['alternative_t_token_ids'] = [(ids + [pad_id] * (pad_a_t_len - len(ids)))[: pad_a_t_len]
                                                 for ids in batch_data['alternative_t_token_ids']]
        return batch_data, pad_p_len, pad_q_len, pad_a_s_len, pad_a_s_len, pad_a_t_len

    def word_iter(self, set_name=None):
        '''
        遍历数据集里的所有词语
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        '''
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['segmented_passage']:
                    yield token
                for token in sample['segmented_query']:
                    yield token
                for tokens in sample['segmented_alternatives']:
                    for token in tokens:
                        yield token

    def convert_to_ids(self, vocab):
        '''
        把原始数据集里的问题和文章转化为id序列
        '''
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['query_token_ids'] = vocab.convert_to_ids(sample['segmented_query'])
                sample['passage_token_ids'] = vocab.convert_to_ids(sample['segmented_passage'])
                sample['alternatives_token_ids'] = []
                for ans in sample['segmented_alternatives']:
                    sample['alternatives_token_ids'].append(vocab.convert_to_ids(ans))

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        '''
        对于任一个指定的数据集(train/dev/test)都通用的batch
        '''
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('没有叫做{}的数据集'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
