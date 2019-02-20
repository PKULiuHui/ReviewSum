# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderDecoder(nn.Module):

    def __init__(self, args, embed):
        super(EncoderDecoder, self).__init__()
        self.name = 'attr2seq'
        self.args = args

        # Word embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # User embedding layer
        self.user_embed = nn.Embedding(args.user_num, args.attr_dim)
        # Product embedding layer
        self.product_embed = nn.Embedding(args.product_num, args.attr_dim)
        # Rating embedding layer
        self.rating_embed = nn.Embedding(10, args.attr_dim)
        # Attributes Encoder
        self.encoder_attr = nn.Linear(3 * args.attr_dim, args.hidden_size * args.num_layers)

        # Decoder
        self.decoder_rnn = nn.GRU(args.embed_dim + args.hidden_size, args.hidden_size, args.num_layers,
                                  batch_first=True, dropout=args.decoder_dropout)
        # Attributes Attention
        self.attention_attr = Attention(args.hidden_size, key_size=args.attr_dim, query_size=args.hidden_size)
        # mix hidden and context into a context_hidden vector
        self.context_hidden = nn.Linear(args.hidden_size + args.attr_dim, args.hidden_size)
        # generate mode layer, context_hidden => word distribution over vocab
        self.generator = nn.Linear(args.hidden_size, args.embed_num, bias=False)

    def decode_step(self, prev_embed, context_hidden, hidden, encoder_attr, proj_key_attr):
        """Perform a single decoder step (1 word)"""

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context_hidden], dim=2)
        output, hidden = self.decoder_rnn(rnn_input, hidden)

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [B, 1, H]
        context_attr, _ = self.attention_attr(query=query, proj_key=proj_key_attr, value=encoder_attr)

        # 计算generate mode下的word distribution，非固定词表部分概率为0
        context_hidden = F.tanh(self.context_hidden(torch.cat([query, context_attr], dim=2)))
        gen_prob = F.log_softmax(self.generator(context_hidden), dim=-1)

        return hidden, context_hidden, gen_prob

    def forward(self, user, product, rating, trg, trg_lengths, test=False):
        # attributes encoder
        user_embed = self.user_embed(user)  # user_embed: [B, A]
        product_embed = self.product_embed(product)  # product_embed: [B, A]
        rating_embed = self.rating_embed(rating)
        attr_final = F.leaky_relu(self.encoder_attr(torch.cat([user_embed, product_embed, rating_embed], dim=-1))).view(
            user_embed.size(0), self.args.num_layers, -1).transpose(0, 1)  # attr_final: [num_layers, B, H]
        encoder_attr = torch.cat([user_embed, product_embed, rating_embed], dim=-1).view(user_embed.size(0), 3, -1)

        trg_embed = self.embed(trg)
        max_len = self.args.sum_max_len
        hidden = attr_final.contiguous()
        context_hidden = hidden[-1].unsqueeze(1)  # context_hidden指融合了context信息的hidden，初始化为hidden[-1]

        # pre-compute projected encoder hidden states(the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key_attr = self.attention_attr.key_layer(encoder_attr)
        pre_output_vectors = []
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            if i == 0:  # <SOS> embedding
                prev_embed = self.embed(torch.LongTensor([1]).cuda()).repeat(len(trg), 1).unsqueeze(1)
            else:
                if not test:  # last trg word embedding
                    prev_embed = trg_embed[:, i - 1].unsqueeze(1)
                else:  # last predicted word embedding
                    prev_embed = self.embed(torch.argmax(pre_output_vectors[-1], dim=-1))
            hidden, context_hidden, word_prob = self.decode_step(prev_embed, context_hidden, hidden, encoder_attr,
                                                                 proj_key_attr)
            pre_output_vectors.append(word_prob)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return pre_output_vectors

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)


class Attention(nn.Module):

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(Attention, self).__init__()
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        # additive attention components, score(hi, hj) = v * tanh(W1 * hi + W2 + hj)
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas  # [B, 1, max_len]

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)  # [B, 1, 2*H]

        # context shape: [B, 1, 2*H], alphas shape: [B, 1, max_len]
        return context, alphas


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(np.matmul(vector_a, vector_b.T))
    denom_a = np.linalg.norm(vector_a)
    denom_b = np.linalg.norm(vector_b)
    cos = num / (denom_a * denom_b)
    sim = 0.5 + 0.5 * cos
    return sim


class myNLLLoss(nn.Module):
    def __init__(self):
        super(myNLLLoss, self).__init__()

    def forward(self, output, target):
        rst = torch.FloatTensor([0]).cuda().squeeze(0)
        for dis, idx in zip(output, target):
            if idx == 0:
                continue
            if dis[idx] == float('-inf'):
                print('error!')
                exit()
            rst -= dis[idx]
        return rst
