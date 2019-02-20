# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0
EOS_IDX = 2


class Beam:
    """
    对一个batch中的每个item都有一个Beam对象，初始化时需要以下参数：
    beam_size：beam个数
    hidden：每一个beam的hidden向量
    context_hidden：每一个beam的context_hidden向量
    word_prob：decoder输入<SOS>后的输出，利用word_prob的前beam_size个词初始化每个beam
    """

    def __init__(self, beam_size, hidden, context_hidden, word_prob):
        self.size = beam_size
        self.seqs = []
        self.seq_done = []
        self.scores = []
        self.hidden = []
        self.context_hidden = []
        self.finish_num = 0
        # 选取word_prob中概率靠前的beam_size个词初始化seqs
        word_prob[3] = float('-inf')
        _, sorted_idx = torch.sort(word_prob, dim=-1, descending=True)
        for i in sorted_idx[:self.size]:
            self.seqs.append([i])
            self.scores.append(word_prob[i])
            self.hidden.append(hidden)
            self.context_hidden.append(context_hidden)
            if i == EOS_IDX:
                self.seq_done.append(True)
            else:
                self.seq_done.append(False)
        for i in range(self.size):
            if self.seq_done[i]:
                self.finish_num += 1
            else:
                break

    """
    判断beam_search是否结束
    """

    def done(self):
        return self.finish_num == self.size

    """
    将beam_size个数据输入到decode_step中后，产生beam_size个word_prob，每个beam选出前beam_size个词分裂成
    beam_size个新beam，从总共的beam_size ** 2个beam中选出得分最高的beam_size个，更新记录。
    word_probs: [beam_size, vocab_size]
    """

    def update(self, word_probs, h, c_h):
        beam_idx = []
        new_words = []
        new_scores = []
        pad = torch.cuda.LongTensor([PAD_IDX]).squeeze(0)
        for i in range(self.size):
            if self.seq_done[i]:
                beam_idx.append(i)
                new_words.append(pad)
                new_scores.append(self.scores[i])
                continue
            word_prob = word_probs[i]
            word_prob[3] = float('-inf')
            _, sorted_idx = torch.sort(word_prob, dim=-1, descending=True)
            for j in sorted_idx[:self.size]:
                beam_idx.append(i)
                new_words.append(j)
                new_scores.append(self.scores[i] + word_prob[j])
        _, sorted_idx = torch.sort(torch.stack(new_scores), descending=True)
        seqs, seq_done, scores, hidden, context_hidden = [], [], [], [], []
        for i in sorted_idx[:self.size]:
            seqs.append(self.seqs[beam_idx[i]] + [new_words[i]])
            if self.seq_done[beam_idx[i]]:
                seq_done.append(True)
            elif new_words[i] == EOS_IDX or new_words[i] == PAD_IDX:
                seq_done.append(True)
            else:
                seq_done.append(False)
            scores.append(new_scores[i])
            hidden.append(h[:, beam_idx[i]])
            context_hidden.append(c_h[beam_idx[i]])
        self.seqs, self.seq_done, self.scores, self.hidden, self.context_hidden = seqs, seq_done, scores, hidden, context_hidden
        self.finish_num = 0
        for i in range(self.size):
            if self.seq_done[i]:
                self.finish_num += 1
            else:
                break
