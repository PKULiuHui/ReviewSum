# coding: utf-8

# Use user/product attributes to generate amazon review summaries(without reviews).
# Ref: https://bastings.github.io/annotated_encoder_decoder/

import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from vocab import Vocab, Dataset
from models import EncoderDecoder, myNLLLoss
from sumeval.metrics.rouge import RougeCalculator

parser = argparse.ArgumentParser(description='attr2seq')
# path info
parser.add_argument('-save_path', type=str, default='checkpoints2/')
parser.add_argument('-embed_path', type=str, default='../../../embedding/glove/glove.review.txt')
parser.add_argument('-train_dir', type=str, default='data/train.txt')
parser.add_argument('-valid_dir', type=str, default='data/dev.txt')
parser.add_argument('-test_dir', type=str, default='data/test.txt')
parser.add_argument('-load_model', type=str, default='')
parser.add_argument('-output_dir', type=str, default='output/')
parser.add_argument('-example_num', type=int, default=4)
# hyper paras
parser.add_argument('-embed_dim', type=int, default=300)
parser.add_argument('-embed_num', type=int, default=0)
parser.add_argument('-word_min_cnt', type=int, default=10)
parser.add_argument('-attr_dim', type=int, default=64)
parser.add_argument('-user_num', type=int, default=0)
parser.add_argument('-product_num', type=int, default=0)
parser.add_argument('-sum_max_len', type=int, default=70)
parser.add_argument('-hidden_size', type=int, default=512)
parser.add_argument('-num_layers', type=int, default=2)
parser.add_argument('-encoder_dropout', type=float, default=0.2)
parser.add_argument('-decoder_dropout', type=float, default=0.2)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-lr_decay', type=float, default=0.5)
parser.add_argument('-max_norm', type=float, default=5.0)
parser.add_argument('-batch_size', type=int, default=50)
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-seed', type=int, default=2333)
parser.add_argument('-valid_every', type=int, default=4000)
parser.add_argument('-test', action='store_true')
parser.add_argument('-use_cuda', type=bool, default=False)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
example_idx = np.random.choice(range(5000), args.example_num)


def adjust_learning_rate(optimizer, index):
    lr = args.lr * (args.lr_decay ** index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(net, criterion, vocab, data_iter, train_next=True):
    net.eval()
    users, products, ratings, reviews, gen_reviews = [], [], [], [], []
    loss, r1, r2, rl = .0, .0, .0, .0
    rouge = RougeCalculator(stopwords=False, lang="en")
    for batch in tqdm(data_iter):
        src_user, src_product, src_rating, trg, trg_lens, trg_text = vocab.read_batch(batch)
        output = net(src_user, src_product, src_rating, trg, trg_lens, test=True)
        output_1 = output.view(-1, output.size(-1))
        trg_output = trg.view(-1)
        loss += criterion(output_1, trg_output).data.item() / len(trg_lens)
        users.extend(src_user.view(-1).data)
        products.extend(src_product.view(-1).data)
        ratings.extend(src_rating.view(-1).data)
        reviews.extend(trg_text)
        output[:, :, 3] = float('-inf')
        rst = torch.argmax(output, dim=-1).tolist()
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
            gen_reviews.append(cur_sum)
            r1 += rouge.rouge_n(cur_sum, trg_text[i], n=1)
            r2 += rouge.rouge_n(cur_sum, trg_text[i], n=2)
            rl += rouge.rouge_l(cur_sum, trg_text[i])
    for i in example_idx:
        print('user: %d product: %d rating: %d' % (users[i], products[i], ratings[i]))
        print('> %s' % reviews[i])
        print('< %s\n' % gen_reviews[i])
    if not train_next:  # 测试阶段将结果写入文件
        with open(args.output_dir + args.load_model, 'w') as f:
            for review, ref, summary in zip(reviews, refs, sums):
                f.write('> %s\n' % review)
                f.write('< %s\n\n' % summary)
    loss /= len(data_iter)
    r1 /= len(gen_reviews)
    r2 /= len(gen_reviews)
    rl /= len(gen_reviews)
    if train_next:
        net.train()
    return loss, r1, r2, rl


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
    with open(args.train_dir, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue
            cur_data = line.split('\t')
            train_data.append(cur_data)
            vocab.add_sentence(cur_data[3].split())
            vocab.add_user(cur_data[0])
            vocab.add_product(cur_data[1])
    with open(args.valid_dir, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue
            cur_data = line.split('\t')
            val_data.append(cur_data)
            vocab.add_sentence(cur_data[3].split())
            vocab.add_user(cur_data[0])
            vocab.add_product(cur_data[1])
    with open(args.test_dir, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue
            cur_data = line.split('\t')
            test_data.append(cur_data)
            vocab.add_sentence(cur_data[3].split())
            vocab.add_user(cur_data[0])
            vocab.add_product(cur_data[1])

    print('Deleting rare words...')
    embed = vocab.trim()
    # save_embed(vocab, '../../embedding/glove/glove.aligned.txt')

    args.embed_num = len(embed)
    args.embed_dim = len(embed[0])
    args.user_num = vocab.user_num
    args.product_num = vocab.product_num

    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_iter = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    net = EncoderDecoder(args, embed)
    if args.use_cuda:
        net.cuda()
    criterion = nn.NLLLoss(ignore_index=vocab.PAD_IDX, size_average=False)
    # criterion = myNLLLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    print('Begin training...')
    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(train_iter):
            src_user, src_product, src_rating, trg, trg_lens, _ = vocab.read_batch(batch)
            output = net(src_user, src_product, src_rating, trg, trg_lens)
            output = output.view(-1, output.size(-1))
            trg_output = trg.view(-1)
            loss = criterion(output, trg_output) / len(trg_lens)
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optim.step()
            optim.zero_grad()
            print('EPOCH [%d/%d]: BATCH_ID=[%d/%d] loss=%f' % (epoch, args.epochs, i, len(train_iter), loss.data))

            cnt = (epoch - 1) * len(train_iter) + i
            if cnt % args.valid_every == 0 and cnt / args.valid_every >= 0:
                print('Begin valid... Epoch %d, Batch %d' % (epoch, i))
                cur_loss, r1, r2, rl = evaluate(net, criterion, vocab, val_iter, True)
                save_path = args.save_path + 'valid_%d_%.4f_%.4f_%.4f_%.4f' % (
                    cnt / args.valid_every, cur_loss, r1, r2, rl)
                net.save(save_path)
                print('Epoch: %2d Cur_Val_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f' %
                      (epoch, cur_loss, r1, r2, rl))

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

    train_data, val_data, test_data = [], [], []
    with open(args.train_dir, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue
            cur_data = line.split('\t')
            train_data.append(cur_data)
            vocab.add_sentence(cur_data[3].split())
            vocab.add_user(cur_data[0])
            vocab.add_product(cur_data[1])
    with open(args.valid_dir, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue
            cur_data = line.split('\t')
            val_data.append(cur_data)
            vocab.add_sentence(cur_data[3].split())
            vocab.add_user(cur_data[0])
            vocab.add_product(cur_data[1])
    with open(args.test_dir, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue
            cur_data = line.split('\t')
            test_data.append(cur_data)
            vocab.add_sentence(cur_data[3].split())
            vocab.add_user(cur_data[0])
            vocab.add_product(cur_data[1])
    embed = vocab.trim()
    args.embed_num = len(embed)
    args.embed_dim = len(embed[0])
    args.user_num = vocab.user_num
    args.product_num = vocab.product_num
    test_dataset = Dataset(val_data)
    test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    print('Loading model...')
    checkpoint = torch.load(args.save_path + args.load_model)
    net = EncoderDecoder(checkpoint['args'], embed)
    net.load_state_dict(checkpoint['model'])
    if args.use_cuda:
        net.cuda()
    criterion = nn.NLLLoss(ignore_index=vocab.PAD_IDX, size_average=False)

    print('Begin testing...')
    loss, r1, r2, rl = evaluate(net, criterion, vocab, test_iter, False)
    print('Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f' % (loss, r1, r2, rl))


def save_embed(vocab, path):
    with open(path, 'w') as f:
        f.write(str(len(vocab.word2id)) + ' ' + str(len(vocab.embed[0])) + '\n')
        for i in range(vocab.word_num):
            f.write(vocab.id2word[i] + ' ' + ' '.join(str(_) for _ in vocab.embed[i]) + '\n')


if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()
