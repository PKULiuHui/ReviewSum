# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRating(nn.Module):

    def __init__(self, args, embed=None):
        super(EncoderRating, self).__init__()
        self.name = 'encoderRate basic model'
        self.args = args

        # Word embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # Text Encoder
        self.encoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers, batch_first=True,
                                  bidirectional=True, dropout=args.dropout)

        # Rating score prediction
        self.rat_predict_1 = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.rat_predict_2 = nn.Linear(args.hidden_size, args.rating_num)

        # User embedding layer
        self.user_embed = nn.Embedding(args.user_num, args.attr_dim)
        # Product embedding layer
        self.product_embed = nn.Embedding(args.product_num, args.attr_dim)

    def forward(self, src, trg, src_, trg_, user, product, rating, vocab_size, u_review, u_sum, p_review, p_sum):
        # useful variables
        batch_size = len(src)
        src_lens = torch.sum(torch.sign(src), dim=1).tolist()

        # embed src words
        src_ = self.embed(src_)

        # feed input to encoder RNN
        packed = pack_padded_sequence(src_, src_lens, batch_first=True)
        encoder_hidden, encoder_final = self.encoder_rnn(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]
        fwd_final = encoder_final[0:encoder_final.size(0):2]
        bwd_final = encoder_final[1:encoder_final.size(0):2]
        encoder_final = torch.cat([fwd_final, bwd_final], dim=2).transpose(0, 1)  # encoder_final: [B, num_layers, 2H]

        # predict rating score
        rat_input = []
        for i in range(batch_size):
            cur_input = torch.sum(encoder_hidden[i][:src_lens[i]], dim=0)
            rat_input.append(cur_input / src_lens[i])
        rat_input = torch.cat(rat_input, dim=-1).view(batch_size, -1)
        rat_output = self.rat_predict_2(F.relu(self.rat_predict_1(rat_input)))
        rat_output = F.log_softmax(rat_output, dim=-1)
        return rat_output

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