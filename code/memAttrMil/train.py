# coding: utf-8

# Use memAttr + word level MIL to generate review summaries.
# 在memAttr模型的基础上，使用word level MIL预测分数，希望得到每个word的情感，然后利用这个信息辅助摘要生成

import os
import json
import argparse
import random
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from vocab import Vocab, Dataset
# from model1 import Model
# from model2 import Model
# from model3 import Model
from model4 import Model
from sumeval.metrics.rouge import RougeCalculator

parser = argparse.ArgumentParser(description='memAttrMil')
parser.add_argument('-save_path', type=str, default='checkpoints4/')
parser.add_argument('-embed_path', type=str, default='../../embedding/glove/glove.aligned.txt')
parser.add_argument('-train_dir', type=str, default='../../data/aligned_memory/train/')
parser.add_argument('-valid_dir', type=str, default='../../data/aligned_memory/valid/')
parser.add_argument('-test_dir', type=str, default='../../data/aligned_memory/test/')
parser.add_argument('-output_dir', type=str, default='output/')
parser.add_argument('-load_model', type=str, default=None)
parser.add_argument('-begin_epoch', type=int, default=1)
parser.add_argument('-example_num', type=int, default=4)
parser.add_argument('-embed_dim', type=int, default=300)
parser.add_argument('-embed_num', type=int, default=0)
parser.add_argument('-word_min_cnt', type=int, default=20)
parser.add_argument('-attr_dim', type=int, default=300)
parser.add_argument('-user_num', type=int, default=0)
parser.add_argument('-product_num', type=int, default=0)
parser.add_argument('-rating_num', type=int, default=6)
parser.add_argument('-review_max_len', type=int, default=300)
parser.add_argument('-sum_max_len', type=int, default=15)
parser.add_argument('-hidden_size', type=int, default=512)
parser.add_argument('-rnn_layers', type=int, default=2)
parser.add_argument('-mem_size', type=int, default=10)
parser.add_argument('-mem_layers', type=int, default=2)
parser.add_argument('-review_encoder_dropout', type=float, default=0.1)
parser.add_argument('-sum_encoder_dropout', type=float, default=0.1)
parser.add_argument('-decoder_dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-lr_decay_ratio', type=float, default=0.5)
parser.add_argument('-lr_decay_start', type=int, default=6)
parser.add_argument('-loss_ratio', type=float, default=1.0)
parser.add_argument('-max_norm', type=float, default=5.0)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-seed', type=int, default=2333)
parser.add_argument('-print_every', type=int, default=10)
parser.add_argument('-valid_every', type=int, default=3000)
parser.add_argument('-test', action='store_true')
parser.add_argument('-test_all', action='store_true')
parser.add_argument('-use_cuda', type=bool, default=False)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
example_idx = np.random.choice(range(5000), args.example_num)


def my_collate(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}


def adjust_learning_rate(optimizer, times):
    lr = args.lr * (args.lr_decay_ratio ** times)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(net, criterion, vocab, data_iter, train_data, train_next=True):
    net.eval()
    reviews = []
    refs = []
    sums = []
    ratings, pre_ratings = [], []
    loss, r1, r2, rl, acc, mae = .0, .0, .0, .0, .0, .0
    rouge = RougeCalculator(stopwords=False, lang="en")
    for batch in tqdm(data_iter):
        src, trg, src_, trg_, src_user, src_product, src_rating, u_review, u_sum, p_review, p_sum, src_text, trg_text \
            = vocab.make_tensors(batch, train_data)
        sum_out, rat_out = net(src, trg, src_, trg_, src_user, src_product, src_rating, vocab.word_num, u_review, u_sum,
                               p_review, p_sum, test=True)
        sum_out_1, rat_out_1 = net(src, trg, src_, trg_, src_user, src_product, src_rating, vocab.word_num, u_review,
                                   u_sum, p_review, p_sum, test=False)
        sum_out_1 = torch.log(sum_out_1.view(-1, sum_out_1.size(-1)) + 1e-20)
        sum_out_gold = trg.view(-1)
        rat_out_gold = src_rating.view(-1)
        sum_loss = criterion(sum_out_1, sum_out_gold).data.item() / len(src)
        rat_loss = criterion(rat_out_1, rat_out_gold).data.item() / len(src)
        loss += sum_loss + args.loss_ratio * rat_loss
        reviews.extend(src_text)
        refs.extend(trg_text)
        sum_out[:, :, 3] = float('-inf')
        rst = torch.argmax(sum_out, dim=-1).tolist()
        rst_rating = torch.argmax(rat_out, dim=-1).tolist()
        rating = src_rating.tolist()
        for i, summary in enumerate(rst):
            cur_sum = ['']
            for idx in summary:
                if idx == vocab.EOS_IDX:
                    break
                w = vocab.id_word(idx)
                cur_sum.append(w)
            cur_sum = ' '.join(cur_sum).strip()
            if len(cur_sum) == 0:
                cur_sum = '<EMP>'
            sums.append(cur_sum)
            ratings.append(rating[i])
            pre_ratings.append(rst_rating[i])
            if rating[i] == rst_rating[i]:
                acc += 1.0
            mae += math.fabs(rst_rating[i] - rating[i])
            r1 += rouge.rouge_n(cur_sum, trg_text[i], n=1)
            r2 += rouge.rouge_n(cur_sum, trg_text[i], n=2)
            rl += rouge.rouge_l(cur_sum, trg_text[i])
    for i in example_idx:
        print('> %s' % reviews[i])
        print('= %d %s' % (ratings[i], refs[i]))
        print('< %d %s\n' % (pre_ratings[i], sums[i]))
    if not train_next:  # 测试阶段将结果写入文件
        with open(args.output_dir + args.load_model, 'w') as f:
            for review, ref, summary in zip(reviews, refs, sums):
                f.write('> %s\n' % review)
                f.write('= %s\n' % ref)
                f.write('< %s\n\n' % summary)
    loss /= len(data_iter)
    r1 /= len(sums)
    r2 /= len(sums)
    rl /= len(sums)
    acc /= len(sums)
    mae /= len(sums)
    if train_next:
        net.train()
    return loss, r1, r2, rl, acc, mae


def train():
    embed = None
    if args.embed_path is not None and os.path.exists(args.embed_path):
        print('Loading pretrained word embedding...')
        embed = {}
        with open(args.embed_path, 'r') as f:
            f.readline()
            for line in f.readlines():
                line = line.strip().split()
                vec = [float(_) for _ in line[1:]]
                embed[line[0]] = vec
    vocab = Vocab(args, embed)

    print('Loading datasets...')
    train_data, val_data, test_data = [], [], []
    fns = os.listdir(args.train_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.train_dir + fn, 'r')
        train_data.append(json.load(f))
        f.close()
        vocab.add_sentence(train_data[-1]['reviewText'].split())
        vocab.add_sentence(train_data[-1]['summary'].split())
        vocab.add_user(train_data[-1]['userID'])
        vocab.add_product(train_data[-1]['productID'])
    fns = os.listdir(args.valid_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()
        vocab.add_sentence(val_data[-1]['reviewText'].split())
        vocab.add_sentence(val_data[-1]['summary'].split())
        vocab.add_user(val_data[-1]['userID'])
        vocab.add_product(val_data[-1]['productID'])
    fns = os.listdir(args.test_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.test_dir + fn, 'r')
        test_data.append(json.load(f))
        f.close()
        vocab.add_sentence(test_data[-1]['reviewText'].split())
        vocab.add_sentence(test_data[-1]['summary'].split())
        vocab.add_user(test_data[-1]['userID'])
        vocab.add_product(test_data[-1]['productID'])

    print('Deleting rare words...')
    embed = vocab.trim()
    args.embed_num = len(embed)
    args.embed_dim = len(embed[0])
    args.user_num = vocab.user_num
    args.product_num = vocab.product_num

    train_dataset = Dataset(train_data)
    val_dataset = Dataset(test_data)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)
    val_iter = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    if args.load_model is not None:
        print('Loading model...')
        checkpoint = torch.load(args.load_model)
        net = Model(checkpoint['args'], embed)
        net.load_state_dict(checkpoint['model'])
    else:
        net = Model(args, embed)
    if args.use_cuda:
        net.cuda()
    net.train()
    criterion = nn.NLLLoss(ignore_index=vocab.PAD_IDX, size_average=False)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    print('Begin training...')
    for epoch in range(args.begin_epoch, args.epochs + 1):
        if epoch >= args.lr_decay_start:
            adjust_learning_rate(optim, epoch - args.lr_decay_start + 1)
        for i, batch in enumerate(train_iter):
            src, trg, src_, trg_, src_user, src_product, src_rating, u_review, u_sum, p_review, p_sum, _1, _2 = \
                vocab.make_tensors(batch, train_data)
            sum_output, rat_output = net(src, trg, src_, trg_, src_user, src_product, src_rating, vocab.word_num,
                                         u_review, u_sum, p_review, p_sum)
            sum_output = torch.log(sum_output.view(-1, sum_output.size(-1)) + 1e-20)
            sum_output_gold = trg.view(-1)
            rat_output_gold = src_rating.view(-1)
            sum_loss = criterion(sum_output, sum_output_gold) / len(src)
            rat_loss = criterion(rat_output, rat_output_gold) / len(src)
            loss = sum_loss + args.loss_ratio * rat_loss
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optim.step()
            optim.zero_grad()

            cnt = (epoch - 1) * len(train_iter) + i
            if cnt % args.print_every == 0:
                print('EPOCH [%d/%d]: BATCH_ID=[%d/%d] sum_loss = %f rat_loss = %f loss=%f' %
                      (epoch, args.epochs, i, len(train_iter), sum_loss.data, rat_loss.data, loss.data))
            if cnt % args.valid_every == 0:
                print('Begin valid... Epoch %d, Batch %d' % (epoch, i))
                cur_loss, r1, r2, rl, acc, mae = evaluate(net, criterion, vocab, val_iter, train_data, True)
                save_path = args.save_path + 'valid_%d_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f' % (
                    cnt / args.valid_every, cur_loss, r1, r2, rl, acc, mae)
                net.save(save_path)
                print('Epoch: %2d Val_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Acc: %f Mae: %f' %
                      (epoch, cur_loss, r1, r2, rl, acc, mae))

    return


def test():
    embed = None
    if args.embed_path is not None and os.path.exists(args.embed_path):
        print('Loading pretrained word embedding...')
        embed = {}
        with open(args.embed_path, 'r') as f:
            f.readline()
            for line in f.readlines():
                line = line.strip().split()
                vec = [float(_) for _ in line[1:]]
                embed[line[0]] = vec
    vocab = Vocab(args, embed)

    print('Loading datasets...')
    train_data, val_data, test_data = [], [], []
    fns = os.listdir(args.train_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.train_dir + fn, 'r')
        train_data.append(json.load(f))
        f.close()
        vocab.add_sentence(train_data[-1]['reviewText'].split())
        vocab.add_sentence(train_data[-1]['summary'].split())
        vocab.add_user(train_data[-1]['userID'])
        vocab.add_product(train_data[-1]['productID'])
    fns = os.listdir(args.valid_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()
        vocab.add_sentence(val_data[-1]['reviewText'].split())
        vocab.add_sentence(val_data[-1]['summary'].split())
        vocab.add_user(val_data[-1]['userID'])
        vocab.add_product(val_data[-1]['productID'])
    fns = os.listdir(args.test_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.test_dir + fn, 'r')
        test_data.append(json.load(f))
        f.close()
        vocab.add_sentence(test_data[-1]['reviewText'].split())
        vocab.add_sentence(test_data[-1]['summary'].split())
        vocab.add_user(test_data[-1]['userID'])
        vocab.add_product(test_data[-1]['productID'])

    print('Deleting rare words...')
    embed = vocab.trim()
    args.embed_num = len(embed)
    args.embed_dim = len(embed[0])
    args.user_num = vocab.user_num
    args.product_num = vocab.product_num
    test_dataset = Dataset(test_data)
    test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    print('Loading model...')
    checkpoint = torch.load(args.save_path + args.load_model)
    net = Model(checkpoint['args'], embed)
    net.load_state_dict(checkpoint['model'])
    if args.use_cuda:
        net.cuda()
    criterion = nn.NLLLoss(ignore_index=vocab.PAD_IDX, size_average=False)

    print('Begin testing...')
    loss, r1, r2, rl, acc, mae = evaluate(net, criterion, vocab, test_iter, train_data, False)
    print('Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Acc: %f Mae: %f' % (loss, r1, r2, rl, acc, mae))


def test_all():
    embed = None
    if args.embed_path is not None and os.path.exists(args.embed_path):
        print('Loading pretrained word embedding...')
        embed = {}
        with open(args.embed_path, 'r') as f:
            f.readline()
            for line in f.readlines():
                line = line.strip().split()
                vec = [float(_) for _ in line[1:]]
                embed[line[0]] = vec
    vocab = Vocab(args, embed)

    print('Loading datasets...')
    train_data, val_data, test_data = [], [], []
    fns = os.listdir(args.train_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.train_dir + fn, 'r')
        train_data.append(json.load(f))
        f.close()
        vocab.add_sentence(train_data[-1]['reviewText'].split())
        vocab.add_sentence(train_data[-1]['summary'].split())
        vocab.add_user(train_data[-1]['userID'])
        vocab.add_product(train_data[-1]['productID'])
    fns = os.listdir(args.valid_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()
        vocab.add_sentence(val_data[-1]['reviewText'].split())
        vocab.add_sentence(val_data[-1]['summary'].split())
        vocab.add_user(val_data[-1]['userID'])
        vocab.add_product(val_data[-1]['productID'])
    fns = os.listdir(args.test_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.test_dir + fn, 'r')
        test_data.append(json.load(f))
        f.close()
        vocab.add_sentence(test_data[-1]['reviewText'].split())
        vocab.add_sentence(test_data[-1]['summary'].split())
        vocab.add_user(test_data[-1]['userID'])
        vocab.add_product(test_data[-1]['productID'])

    print('Deleting rare words...')
    embed = vocab.trim()
    args.embed_num = len(embed)
    args.embed_dim = len(embed[0])
    args.user_num = vocab.user_num
    args.product_num = vocab.product_num
    test_dataset = Dataset(test_data)
    test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    start, end = 17, 33
    fns = os.listdir(args.save_path)
    f = open(args.save_path + 'result', 'w')
    for idx in range(start, end):
        fn = None
        for name in fns:
            if 'valid_%d_' % idx in name:
                fn = name
                break
        if fn is None:
            continue
        checkpoint = torch.load(args.save_path + fn)
        net = Model(checkpoint['args'], embed)
        net.load_state_dict(checkpoint['model'])
        if args.use_cuda:
            net.cuda()
        criterion = nn.NLLLoss(ignore_index=vocab.PAD_IDX, size_average=False)
        loss, r1, r2, rl, acc, mae = evaluate(net, criterion, vocab, test_iter, train_data, True)
        f.write('Idx: %d Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Acc: %f Mae: %f\n' % (
            idx, loss, r1, r2, rl, acc, mae))


if __name__ == '__main__':
    if args.test:
        test()
    elif args.test_all:
        test_all()
    else:
        train()
