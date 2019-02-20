# coding: utf-8

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
from sumeval.metrics.rouge import RougeCalculator


# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, emb, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed = emb
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, test=False):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask, test=test)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.embed(src), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None, test=False):
        return self.decoder(self.embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden, test=test, emb=self.embed)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden=None, max_len=None, test=False, emb=None):
        """Unroll the decoder one step at a time."""
        assert emb is not None
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            if i == 0:  # <SOS> embedding
                prev_embed = emb(torch.LongTensor([1]).cuda()).repeat(len(src_mask), 1).unsqueeze(1)
            else:
                if not test:  # last trg word embedding
                    prev_embed = trg_embed[:, i - 1].unsqueeze(1)
                else:  # last predicted word embedding
                    prev_idx = torch.argmax(pre_output_vectors[-1], dim=-1)
                    prev_embed = emb(prev_idx)
            # prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        mask = mask.unsqueeze(1)
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


def make_model(emb, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)
    emb_size = emb.size(1)
    e = nn.Embedding(emb.size(0), emb.size(1))
    e.weight.data.copy_(emb)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        e, Generator(hidden_size, emb.size(0)))

    return model.cuda() if USE_CUDA else model


class Vocab:
    def __init__(self, embedding=None):
        self.pretrained_embed = embedding
        self.embed_dim = 300
        self.word_min_cnt = 20
        self.embed = []  # pretrained embed是embed的一部分
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        self.next_idx = 4
        self.word2id = {self.PAD_TOKEN: self.PAD_IDX, self.SOS_TOKEN: self.SOS_IDX, self.EOS_TOKEN: self.EOS_IDX,
                        self.UNK_TOKEN: self.UNK_IDX}
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.word2cnt = {self.PAD_TOKEN: 10000, self.SOS_TOKEN: 10000, self.EOS_TOKEN: 10000,
                         self.UNK_TOKEN: 10000}  # 给这几个标记较大的值防止后面被删去
        for i in range(4):
            self.embed.append(np.random.normal(size=self.embed_dim))
        if embedding is not None:
            for w in sorted(embedding.keys()):
                if w not in self.word2id:
                    self.word2id[w] = self.next_idx
                    self.id2word[self.next_idx] = w
                    self.word2cnt[w] = 0
                    self.embed.append(embedding[w])
                    self.next_idx += 1

    # 读取一个句子，更新词表以及相关记录
    def add_sentence(self, sent):
        for w in sent:
            if w not in self.word2id:
                self.word2id[w] = self.next_idx
                self.id2word[self.next_idx] = w
                self.word2cnt[w] = 1
                self.embed.append(np.random.normal(size=self.embed_dim))
                self.next_idx += 1
            else:
                self.word2cnt[w] += 1

    # 已经读完了所有句子，对词表进行剪枝，只保留出现次数大于等于20的词以及相应词向量
    def trim(self):
        print('original vocab size: %d' % len(self.word2cnt))
        reserved_words, reserved_idx = [], []
        for i in range(self.next_idx):
            w = self.id2word[i]
            if self.word2cnt[w] >= self.word_min_cnt:
                reserved_words.append(w)
                reserved_idx.append(i)
        cnt = 0
        word2id, id2word, word2cnt = {}, {}, {}
        embed1 = []
        for w in reserved_words:
            word2id[w] = cnt
            id2word[cnt] = w
            word2cnt[w] = self.word2cnt[w]
            cnt += 1
        for i in reserved_idx:
            embed1.append(self.embed[i])
        assert len(word2id) == len(id2word) and len(id2word) == len(word2cnt) and len(word2cnt) == len(embed1)
        self.word2id, self.id2word, self.word2cnt, self.embed = word2id, id2word, word2cnt, embed1
        print('Vocab size: %d' % len(self.word2id))
        embed1 = torch.FloatTensor(embed1)
        embed1 = embed1.cuda()
        return embed1

    def word_id(self, w):
        if w in self.word2id:
            return self.word2id[w]
        return self.UNK_IDX

    def id_word(self, idx):
        assert 0 <= idx < len(self.id2word)
        return self.id2word[idx]

    # 给定一个batch的文本数据，生成神经网络所需要的input tensors
    def make_tensors(self, batch):
        src_text, trg_text = [], []  # 存储评论和摘要的文本，便于valid时查看效果，按照评论长度由大到小排序
        for review, summary in zip(batch['reviewText'], batch['summary']):
            src_text.append(review)
            trg_text.append(summary)
        idx = list(range(len(batch['summary'])))  # 将一个batch中的数据由长到短排序，方便作为GRU输入
        idx.sort(key=lambda k: len(src_text[k].split()), reverse=True)
        src_text = [src_text[i] for i in idx]
        trg_text = [trg_text[i] for i in idx]

        src_max_len = len(src_text[0].split())
        trg_max_len = 15
        src, trg, src_mask, trg_mask, src_lens, trg_lens = [], [], [], [], [], []
        for review in src_text:
            review = review.split()[:src_max_len]
            cur_idx = []
            for w in review:
                cur_idx.append(self.word_id(w))
            cur_idx.extend([self.PAD_IDX] * (src_max_len - len(review)))
            src.append(cur_idx)
            src_mask.append([1] * len(review) + [0] * (src_max_len - len(review)))
            src_lens.append(len(review))
        for summary in trg_text:
            summary = summary.split() + [self.EOS_TOKEN]
            summary = summary[:trg_max_len]
            cur_idx = []
            for w in summary:
                cur_idx.append(self.word_id(w))
            cur_idx.extend([self.PAD_IDX] * (trg_max_len - len(summary)))
            trg.append(cur_idx)
            trg_mask.append([1] * len(summary) + [0] * (trg_max_len - len(summary)))
            trg_lens.append(len(summary))
        src, trg, src_mask, trg_mask = torch.LongTensor(src), torch.LongTensor(trg), torch.LongTensor(
            src_mask), torch.LongTensor(trg_mask)
        src, trg, src_mask, trg_mask = src.cuda(), trg.cuda(), src_mask.cuda(), trg_mask.cuda()

        return src, trg, src_mask, trg_mask, src_lens, trg_lens, src_text, trg_text


print('Loading pretrained word embedding...')
embed = {}
with open('../embedding/glove/glove.unaligned.txt', 'r') as f:
    f.readline()
    for line in f.readlines():
        line = line.strip().split()
        vec = [float(_) for _ in line[1:]]
        embed[line[0]] = vec
vocab = Vocab(embed)

print('Loading datasets...')
train_dir = '../data/unaligned_dense/train/'
valid_dir = '../data/unaligned_dense/valid/'
test_dir = '../data/unaligned_dense/test/'
train_data, val_data, test_data = [], [], []
fns = os.listdir(train_dir)
fns.sort(key=lambda p: int(p.split('.')[0]))
for fn in tqdm(fns):
    f = open(train_dir + fn, 'r')
    train_data.append(json.load(f))
    f.close()
    vocab.add_sentence(train_data[-1]['reviewText'].split())
    vocab.add_sentence(train_data[-1]['summary'].split())
fns = os.listdir(valid_dir)
fns.sort(key=lambda p: int(p.split('.')[0]))
for fn in tqdm(fns):
    f = open(valid_dir + fn, 'r')
    val_data.append(json.load(f))
    f.close()
    vocab.add_sentence(val_data[-1]['reviewText'].split())
    vocab.add_sentence(val_data[-1]['summary'].split())
fns = os.listdir(test_dir)
fns.sort(key=lambda p: int(p.split('.')[0]))
for fn in tqdm(fns):
    f = open(test_dir + fn, 'r')
    test_data.append(json.load(f))
    f.close()
    vocab.add_sentence(test_data[-1]['reviewText'].split())
    vocab.add_sentence(test_data[-1]['summary'].split())

print('Deleting rare words...')
embed = vocab.trim()


class Dataset(data.Dataset):
    def __init__(self, examples):
        super(Dataset, self).__init__()
        self.examples = examples
        self.training = False

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex

    def __len__(self):
        return len(self.examples)


train_dataset = Dataset(train_data)
val_dataset = Dataset(val_data)
train_iter = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_iter = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
example_idx = np.random.choice(range(5000), 4)


def evaluate(net, criterion, vocab, data_iter):
    reviews = []
    refs = []
    sums = []
    loss, r1, r2, rl = .0, .0, .0, .0
    rouge = RougeCalculator(stopwords=False, lang="en")
    for batch in tqdm(data_iter):
        src, trg, src_mask, trg_mask, src_lens, trg_lens, src_text, trg_text = vocab.make_tensors(batch)
        out, _, pre = net.forward(src, trg, src_mask, trg_mask, src_lens, trg_lens, test=True)
        pre = model.generator(pre)
        pre_output = pre.view(-1, pre.size(-1))
        trg_output = trg.view(-1)
        loss += criterion(pre_output, trg_output).data.item() / len(src_lens)
        reviews.extend(src_text)
        refs.extend(trg_text)
        pre[:, :, 3] = float('-inf')
        rst = torch.argmax(pre, dim=-1).tolist()
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
            r1 += rouge.rouge_n(cur_sum, trg_text[i], n=1)
            r2 += rouge.rouge_n(cur_sum, trg_text[i], n=2)
            rl += rouge.rouge_l(cur_sum, trg_text[i])
    for i in example_idx:
        print('> %s' % reviews[i])
        print('= %s' % refs[i])
        print('< %s\n' % sums[i])
    loss /= len(data_iter)
    r1 /= len(sums)
    r2 /= len(sums)
    rl /= len(sums)
    return loss, r1, r2, rl


def train(model, num_epochs=10, lr=0.0003, print_every=10, record_every=1000, record_fn='record.txt'):
    if USE_CUDA:
        model.cuda()
    if os.path.exists(record_fn):
        os.remove(record_fn)

    criterion = nn.NLLLoss(size_average=False, ignore_index=vocab.PAD_IDX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_compute = SimpleLossCompute(model.generator, criterion, optim)

    dev_perplexities = []

    for epoch in range(num_epochs):
        print("Epoch", epoch)
        model.train()

        for i, batch in enumerate(train_iter):
            src, trg, src_mask, trg_mask, src_lens, trg_lens, _1, _2 = vocab.make_tensors(batch)
            out, _, pre_output = model.forward(src, trg, src_mask, trg_mask, src_lens, trg_lens)
            loss = loss_compute(pre_output, trg, float(len(src_lens)))

            cnt = epoch * len(train_iter) + i

            if cnt % print_every == 0:
                print("Epoch: %d Batch: %d Loss: %f" % (epoch, i, loss))

            if cnt % record_every == 0:
                model.eval()
                with torch.no_grad():
                    print('Begin valid... Epoch %d, Batch %d' % (epoch, i))
                    cur_loss, r1, r2, rl = evaluate(model, criterion, vocab, val_iter)
                    print('Epoch: %2d Cur_Val_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f' %
                          (epoch, cur_loss, r1, r2, rl))
                    with open(record_fn, 'a') as f:
                        f.write('Epoch: %2d Cur_Val_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f\n' % (
                            epoch, cur_loss, r1, r2, rl))
                model.train()

    return dev_perplexities


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item()


model = make_model(embed, hidden_size=512, num_layers=2, dropout=0.2)
dev_perplexities = train(model, print_every=10, record_every=500, record_fn='record.txt')
