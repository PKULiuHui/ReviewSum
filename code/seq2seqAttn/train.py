# coding: utf-8

# Use seq2seq + Bahdanau attention to generate amazon review summaries.
# Ref: https://bastings.github.io/annotated_encoder_decoder/

import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from vocab import Vocab
from dataset import Dataset
from models import Encoder, Decoder, EncoderDecoder, BahdanauAttention, Generator
import sys

sys.path.append('../')
from myrouge.rouge import get_rouge_score

parser = argparse.ArgumentParser(description='seq2seqAttn')
# path info
parser.add_argument('-save_path', type=str, default='checkpoints/')
parser.add_argument('-embed_path', type=str, default='../../embedding/numberbatch-en-17.06.txt')
parser.add_argument('-train_dir', type=str, default='../../data/user_based/train/')
parser.add_argument('-valid_dir', type=str, default='../../data/user_based/valid/')
parser.add_argument('-test_dir', type=str, default='../../data/user_based/test/')
parser.add_argument('-output_dir', type=str, default='output/')
# hyper paras
parser.add_argument('-embed_dim', type=int, default=300)
parser.add_argument('-embed_num', type=int, default=0)
parser.add_argument('-word_min_cnt', type=int, default=20)
parser.add_argument('-sum_max_len', type=int, default=15)
parser.add_argument('-hidden_size', type=int, default=512)
parser.add_argument('-num_layers', type=int, default=2)
parser.add_argument('-encoder_dropout', type=float, default=.0)
parser.add_argument('-decoder_dropout', type=float, default=.0)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-max_norm', type=float, default=5.0)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-epochs', type=int, default=8)
parser.add_argument('-seed', type=int, default=2333)
parser.add_argument('-valid_every', type=int, default=1000)
parser.add_argument('-use_cuda', type=bool, default=False)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


def evaluate(net, criterion, vocab, val_iter, train_next):
    return


def train():
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
    for fn in tqdm(os.listdir(args.valid_dir)):
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()
        vocab.add_sentence(val_data[-1]['reviewText'].split())
        vocab.add_sentence(val_data[-1]['summary'].split())
    for fn in tqdm(os.listdir(args.test_dir)):
        f = open(args.test_dir + fn, 'r')
        test_data.append(json.load(f))
        f.close()
        vocab.add_sentence(test_data[-1]['reviewText'].split())
        vocab.add_sentence(test_data[-1]['summary'].split())

    print('Deleting rare words...')
    embed = vocab.trim()
    vocab.show_info()
    args.embed_num = len(embed)
    args.embed_dim = len(embed[0])

    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    val_iter = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    attention = BahdanauAttention(args.hidden_size)
    net = EncoderDecoder(args, Encoder(args), Decoder(args, attention), embed, Generator(args))
    if args.use_cuda:
        net.cuda()
    criterion = nn.NLLLoss(ignore_index=vocab.PAD_IDX)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    print('Begin training...')
    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(train_iter):
            src, trg, src_mask, src_lens, trg_lens, _1, _2 = vocab.make_tensors(batch)
            pre_output = net(src, trg, src_mask, src_lens, trg_lens)
            pre_output = pre_output.view(-1, pre_output.size(-1))
            trg_output = trg.view(-1)
            loss = criterion(pre_output, trg_output)
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optim.step()
            optim.zero_grad()
            print('EPOCH [%d/%d]: BATCH_ID=[%d/%d] loss=%f' % (epoch, args.epochs, i, len(train_iter), loss.data))

            cnt = (epoch - 1) * len(train_data) + i
            if cnt % args.valid_every == 0 and cnt / args.valid_every > 0:
                print('Begin valid... Epoch %d, Batch %d' % (epoch, i))
                cur_loss, r1, r2, rl, rsu = evaluate(net, criterion, vocab, val_iter, True)
                save_path = args.save_dir + args.model + '_%d_%.4f_%.4f_%.4f_%.4f_%.4f' % (
                    cnt / args.valid_every, cur_loss, r1, r2, rl, rsu)
                net.save(save_path)
                print('Epoch: %2d Cur_Val_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f Rouge-SU4: %f' %
                      (epoch, cur_loss, r1, r2, rl, rsu))
    return


if __name__ == '__main__':
    train()
