# coding=utf-8
import torch
import numpy as np
import torch.utils.data as data


# Vocab: composed of two parts
# Part 1: fixed part, composed of popular words
# Part 2: changeable part, composed of words appearing in current batch but not in fixed vocab
class Vocab:
    def __init__(self, args, embed=None):
        self.args = args
        self.embed = []
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        self.word_num = 4
        self.fixed_num = 4
        self.word2id = {self.PAD_TOKEN: self.PAD_IDX, self.SOS_TOKEN: self.SOS_IDX, self.EOS_TOKEN: self.EOS_IDX,
                        self.UNK_TOKEN: self.UNK_IDX}
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.word2cnt = {self.PAD_TOKEN: 10000, self.SOS_TOKEN: 10000, self.EOS_TOKEN: 10000,
                         self.UNK_TOKEN: 10000}
        self.user_num = 1
        self.user2id = {'<UNK-USER>': 0}
        self.id2user = {0: '<UNK-USER>'}
        self.user2cnt = {'<UNK-USER>': 10000}
        self.product_num = 1
        self.product2id = {'<UNK-PRODUCT>': 0}
        self.id2product = {0: '<UNK-PRODUCT>'}
        self.product2cnt = {'<UNK-PRODUCT>': 10000}
        for i in range(4):
            self.embed.append(np.random.normal(size=args.embed_dim))
        if embed is not None:
            for w in sorted(embed.keys()):
                if w not in self.word2id:
                    self.word2id[w] = self.word_num
                    self.id2word[self.word_num] = w
                    self.word2cnt[w] = 0
                    self.embed.append(embed[w])
                    self.word_num += 1
        self.fixed_num = self.word_num

    def add_sentence(self, sent):
        for w in sent:
            if w not in self.word2id:
                self.word2id[w] = self.word_num
                self.id2word[self.word_num] = w
                self.word2cnt[w] = 1
                self.embed.append(np.random.normal(size=self.args.embed_dim))
                self.word_num += 1
            else:
                self.word2cnt[w] += 1
        self.fixed_num = self.word_num

    def add_user(self, user):
        if user not in self.user2id:
            self.user2id[user] = self.user_num
            self.id2user[self.user_num] = user
            self.user2cnt[user] = 1
            self.user_num += 1
        else:
            self.user2cnt[user] += 1

    def add_product(self, product):
        if product not in self.product2id:
            self.product2id[product] = self.product_num
            self.id2product[self.product_num] = product
            self.product2cnt[product] = 1
            self.product_num += 1
        else:
            self.product2cnt[product] += 1

    def trim(self):
        print('original vocab size: %d' % len(self.word2cnt))
        reserved_words, reserved_idx = [], []
        for i in range(self.word_num):
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
        self.word_num = cnt
        self.fixed_num = cnt
        assert len(word2id) == len(id2word) and len(id2word) == len(word2cnt) and len(word2cnt) == len(embed)
        self.word2id, self.id2word, self.word2cnt, self.embed = word2id, id2word, word2cnt, embed
        print('Vocab size: %d' % len(self.word2id))

        print('original user num = %d, product num = %d' % (self.user_num, self.product_num))
        reserved_user, reserved_product = [], []
        for i in range(self.user_num):
            u = self.id2user[i]
            if self.user2cnt[u] >= 10:
                reserved_user.append(u)
        cnt = 0
        user2id, id2user, user2cnt = {}, {}, {}
        for u in reserved_user:
            user2id[u] = cnt
            id2user[cnt] = u
            user2cnt[u] = self.user2cnt[u]
            cnt += 1
        self.user_num = cnt
        self.user2id, self.id2user, self.user2cnt = user2id, id2user, user2cnt

        for i in range(self.product_num):
            p = self.id2product[i]
            if self.product2cnt[p] >= 10:
                reserved_product.append(p)
        cnt = 0
        product2id, id2product, product2cnt = {}, {}, {}
        for p in reserved_product:
            product2id[p] = cnt
            id2product[cnt] = p
            product2cnt[p] = self.product2cnt[p]
            cnt += 1
        self.product_num = cnt
        self.product2id, self.id2product, self.product2cnt = product2id, id2product, product2cnt
        print('user num = %d, product num = %d' % (self.user_num, self.product_num))

        for i in range(self.user_num):
            user = self.id2user[i]
            self.word2id[user] = self.fixed_num
            self.id2word[self.fixed_num] = user
            self.word2cnt[user] = 10000
            self.embed.append(np.random.normal(size=self.args.embed_dim))
            self.fixed_num += 1
        for i in range(self.product_num):
            pro = self.id2product[i]
            self.word2id[pro] = self.fixed_num
            self.id2word[self.fixed_num] = pro
            self.word2cnt[pro] = 10000
            self.embed.append(np.random.normal(size=self.args.embed_dim))
            self.fixed_num += 1
        self.word_num = self.fixed_num
        self.embed = torch.FloatTensor(self.embed)
        if self.args.use_cuda:
            self.embed = self.embed.cuda()
        return self.embed

    def word_id(self, w):
        if w in self.word2id:
            return self.word2id[w]
        return self.UNK_IDX

    def id_word(self, idx):
        assert 0 <= idx < self.word_num
        return self.id2word[idx]

    # generate tensors for a batch
    def read_batch(self, batch):
        # delete the last changeable vocab part
        for i in range(self.fixed_num, self.word_num):
            w = self.id_word(i)
            self.id2word.pop(i)
            self.word2id.pop(w)
        self.word_num = self.fixed_num

        src_text, trg_text = [], []
        for review, summary in zip(batch['reviewText'], batch['summary']):
            src_text.append(review)
            trg_text.append(summary)
        idx = list(range(len(batch['summary'])))
        idx.sort(key=lambda k: len(src_text[k].split()), reverse=True)
        src_text = [src_text[i] for i in idx]
        trg_text = [trg_text[i] for i in idx]

        # generate changeable part for current batch
        for review in src_text:
            review = review.split()
            for w in review:
                if w not in self.word2id:
                    self.word2id[w] = self.word_num
                    self.id2word[self.word_num] = w
                    self.word_num += 1

        src_max_len = len(src_text[0].split()) + 2
        trg_max_len = self.args.sum_max_len
        src, trg, src_mask, src_lens, trg_lens = [], [], [], [], []
        src_embed, trg_embed = [], []  # 针对embed的src和trg输入，与src和trg相比，将处于可变词典的词序号替换成UNK_IDX
        for i, review in zip(idx, src_text):
            user, product = batch['userID'][i], batch['productID'][i]
            assert user in self.user2id and product in self.product2id
            user = user if user in self.user2id else '<UNK-USER>'
            product = product if product in self.product2id else '<UNK-PRODUCT>'
            review = [user, product] + review.split()[:src_max_len - 2]
            cur_idx = []
            for w in review:
                cur_idx.append(self.word_id(w))
            cur_idx.extend([self.PAD_IDX] * (src_max_len - len(review)))
            src.append(cur_idx)
            src_embed.append([i if i < self.fixed_num else self.UNK_IDX for i in cur_idx])
            src_mask.append([1] * len(review) + [0] * (src_max_len - len(review)))
            src_lens.append(len(review))
        for i, summary in enumerate(trg_text):
            summary = summary.split() + [self.EOS_TOKEN]
            summary = summary[:trg_max_len]
            cur_idx = []
            for w in summary:
                cur_id = self.word_id(w)
                if cur_id >= self.fixed_num and cur_id not in src[i]:
                    cur_id = self.UNK_IDX
                cur_idx.append(cur_id)
            cur_idx.extend([self.PAD_IDX] * (trg_max_len - len(summary)))
            trg.append(cur_idx)
            trg_embed.append([i if i < self.fixed_num else self.UNK_IDX for i in cur_idx])
            trg_lens.append(len(summary))

        src, trg, src_mask = torch.LongTensor(src), torch.LongTensor(trg), torch.LongTensor(src_mask)
        src_embed, trg_embed = torch.LongTensor(src_embed), torch.LongTensor(trg_embed)
        if self.args.use_cuda:
            src, trg, src_mask = src.cuda(), trg.cuda(), src_mask.cuda()
            src_embed, trg_embed = src_embed.cuda(), trg_embed.cuda()

        return src, trg, src_embed, trg_embed, src_mask, src_lens, trg_lens, src_text, trg_text


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
