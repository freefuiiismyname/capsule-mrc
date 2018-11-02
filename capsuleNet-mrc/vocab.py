# -*- coding:utf8 -*-
"""
This module implements the Vocab class for converting string to id and back
"""

import numpy as np
from gensim.models import Word2Vec


class Vocab(object):
    """
    通过tokens一致的嵌入，实现一个词汇表来存储数据里的tokens（词语）
    """

    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        获取词汇表的大小
        """
        return len(self.id2token)

    def load_from_file(self, file_path):
        """
        从文件路径加载词汇表
        """
        for line in open(file_path, 'r', encoding='utf-8'):
            token = line.rstrip('\n')
            self.add(token)

    def get_id(self, token):
        """
        获得某个词语的id，如果词语不在词汇表中，则返回未知词标识
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        """
        获取与id相对应的词语，如果id不在词汇表中，返回未知标识
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        """
        把词语加入到词汇表
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        """
        过滤掉一些低频词
        """
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def randomly_init_embeddings(self, embed_dim):
        """
        随机初始化词向量
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    def load_pretrained_embeddings(self, embedding_path):
        """
        根据文件路径，加载预训练的词向量
        不在该词向量集中的词语将被过滤
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        print('load embedding path {}'.format(embedding_path))
        model = Word2Vec.load(embedding_path)
        trained_embeddings = {}
        for token in model.wv.vocab:
            if token not in self.token2id:
                continue
            trained_embeddings[token] = model.wv[token]
            if self.embed_dim is None:
                self.embed_dim = len(model.wv[token])

        filtered_tokens = trained_embeddings.keys()
        # 重构词语和id的映射关系
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # 加载词嵌入
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        count = 0
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]
                count +=1
        print('词向量初始化个数{}'.format(count))

    def convert_to_ids(self, tokens):
        """
        将一组词语转化为id序列
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        将一组id序列转化为一组词语
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
