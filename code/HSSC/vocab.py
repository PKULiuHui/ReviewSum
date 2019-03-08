# coding=utf-8
import torch
import numpy as np


# Vocab：记录词典（词典由pretrained embed和数据集中的常见词组成）;记录初始embedding;将一个batch的数据转成tensor
class Vocab:
    def __init__(self, args, embed=None):
        self.args = args
        self.pretrained_embed = embed
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
            self.embed.append(np.random.normal(size=args.embed_dim))
        if embed is not None:
            for w in sorted(embed.keys()):
                if w not in self.word2id:
                    self.word2id[w] = self.next_idx
                    self.id2word[self.next_idx] = w
                    self.word2cnt[w] = 0
                    self.embed.append(embed[w])
                    self.next_idx += 1

    # 读取一个句子，更新词表以及相关记录
    def add_sentence(self, sent):
        for w in sent:
            if w not in self.word2id:
                self.word2id[w] = self.next_idx
                self.id2word[self.next_idx] = w
                self.word2cnt[w] = 1
                self.embed.append(np.random.normal(size=self.args.embed_dim))
                self.next_idx += 1
            else:
                self.word2cnt[w] += 1

    # 已经读完了所有句子，对词表进行剪枝，只保留出现次数大于等于self.args.word_min_cnt的词以及相应词向量
    def trim(self):
        print('original vocab size: %d' % len(self.word2cnt))
        reserved_words, reserved_idx = [], []
        for i in range(self.next_idx):
            w = self.id2word[i]
            if self.word2cnt[w] >= self.args.word_min_cnt:
                reserved_words.append(w)
                reserved_idx.append(i)
        cnt = 0
        word2id, id2word, word2cnt = {}, {}, {}
        embed = []
        for w in reserved_words:
            word2id[w] = cnt
            id2word[cnt] = w
            word2cnt[w] = self.word2cnt[w]
            cnt += 1
        for i in reserved_idx:
            embed.append(self.embed[i])
        assert len(word2id) == len(id2word) and len(id2word) == len(word2cnt) and len(word2cnt) == len(embed)
        self.word2id, self.id2word, self.word2cnt, self.embed = word2id, id2word, word2cnt, embed
        print('Vocab size: %d' % len(self.word2id))
        embed = torch.FloatTensor(embed)
        if self.args.use_cuda:
            embed = embed.cuda()
        return embed

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
        trg_max_len = self.args.sum_max_len
        src, trg, src_mask, src_lens, trg_lens = [], [], [], [], []
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
            trg_lens.append(len(summary))

        rating = []
        for i in idx:
            cur_rating = int(batch['rating'][i]) - 1
            assert cur_rating >= 0
            rating.append(cur_rating)
        src, trg, src_mask = torch.LongTensor(src), torch.LongTensor(trg), torch.LongTensor(src_mask)
        rating = torch.LongTensor(rating)
        if self.args.use_cuda:
            src, trg, src_mask, rating = src.cuda(), trg.cuda(), src_mask.cuda(), rating.cuda()

        return src, trg, src_mask, src_lens, trg_lens, rating, src_text, trg_text
