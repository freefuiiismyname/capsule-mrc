# -*- coding:utf8 -*-

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import BRCDataset
from vocab import Vocab
from rc_model import RCModel
import rc_model as rcmodel


def parse_args():
    """
    解析命令行变量
    """
    parser = argparse.ArgumentParser('Reading Comprehension on aic dataset')
    parser.add_argument('--prepare', action='store_true', default=False,
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true', default=True,
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='3',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.0005,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=25,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--load_epoch', default=1)
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=30,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=10,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/trainset/train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/devset/dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/testset/test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path', default='../data/logging2',
                               help='path of the log file. If not set, logs are printed to console')

    return parser.parse_args()


def prepare(args):
    """
    检查数据，创建目录，准备词汇表和词嵌入
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('检查数据文件...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} 文件不存在.'.format(data_path)
    logger.info('建立目录...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('创建词汇表...')
    brc_data = BRCDataset(args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    vocab = Vocab(lower=True)
    print(vocab.size())
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)

    print(vocab.size())
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('过滤掉 {} 个词语, 最终的词汇量是 {}'.format(filtered_num,
                                                vocab.size()))
    logger.info('指定词向量...')
    # vocab.randomly_init_embeddings(args.embed_size)
    vocab.load_pretrained_embeddings('../data/vocab/word2vec.model')

    print(vocab.size())
    logger.info('保存词汇表...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('完成预备过程!')


def train(args):
    """
    训练阅读理解模型
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("brc")

    file_handler = logging.FileHandler(args.log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(args)

    logger.info('加载数据集和词汇表...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    logger.info('词语转化为id序列...')
    brc_data.convert_to_ids(vocab)
    logger.info('初始化模型...')
    rc_model = RCModel(vocab, args)
    logger.info('训练模型...')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('训练完成!')


def evaluate(args):
    """
    对训练好的模型进行验证
    """
    logger = logging.getLogger("brc")
    logger.info('加载数据集和词汇表...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, '找不到验证文件.'
    brc_data = BRCDataset(args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    logger.info('把文本转化为id序列...')
    brc_data.convert_to_ids(vocab)
    logger.info('重载模型...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo + '_' + str(args.load_epoch))
    logger.info('验证模型...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('验证集上的损失为: {}'.format(dev_loss))
    logger.info('预测的答案证保存到 {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    预测测试文件的答案
    """
    logger = logging.getLogger("brc")
    logger.info('加载数据集和词汇表...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, '找不到测试文件.'
    brc_data = BRCDataset(args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('把文本转化为id序列...')
    brc_data.convert_to_ids(vocab)
    logger.info('重载模型...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo + '_' + str(args.load_epoch))
    logger.info('预测测试集的答案...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    预训练并运行整个系统.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    run()
