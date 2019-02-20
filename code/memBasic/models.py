# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MemBasic(nn.Module):

    def __init__(self, args, embed):
        super(MemBasic, self).__init__()
        self.name = 'seq2seqAttn + memory'
        self.args = args
        # Embedding layer, shared by all encoders and decoder
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # Review Encoder
        self.review_encoder = nn.GRU(args.embed_dim, args.hidden_size, args.rnn_layers, batch_first=True,
                                     bidirectional=True, dropout=args.review_encoder_dropout)
        # Summary Encoder
        self.sum_encoder = nn.GRU(args.embed_dim, args.hidden_size, args.rnn_layers, batch_first=True,
                                  bidirectional=True, dropout=args.sum_encoder_dropout)
        # Memory Scoring layer
        self.key_score = nn.Linear(3, 1)  # [u : p : ri·rj] => similarity_score
        self.new_query = nn.Linear(4 * args.hidden_size, 2 * args.hidden_size)  # [query : mem_out] => new_query
        # Attention
        self.attention = Attention(args.hidden_size)
        # Decoder
        self.decoder_rnn = nn.GRU(args.embed_dim + args.hidden_size + 2 * args.hidden_size, args.hidden_size,
                                  args.rnn_layers, batch_first=True, dropout=args.decoder_dropout)
        self.init_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)
        self.context_hidden = nn.Linear(3 * args.hidden_size, args.hidden_size, bias=False)
        self.generator = nn.Linear(args.hidden_size, args.embed_num, bias=False)

    def decode_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden, context_hidden):
        """Perform a single decoder step (1 word)"""

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context_hidden], dim=2)
        output, hidden = self.decoder_rnn(rnn_input, hidden)

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [B, 1, H]
        context, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)
        context_hidden = F.tanh(self.context_hidden(torch.cat([query, context], dim=2)))
        pre_output = self.dropout_layer(context_hidden)
        pre_output = self.generator(pre_output)

        return hidden, context_hidden, pre_output

    def forward(self, src, trg, mem_up, mem_review, mem_sum, test=False):
        # useful variables
        batch_size = len(src)
        src_lens = torch.sum(torch.sign(src), dim=1).data
        src_mask = torch.sign(src)
        mem_review_lens = torch.sum(torch.sign(mem_review), dim=1).data
        mem_sum_lens = torch.sum(torch.sign(mem_sum), dim=1).data

        # TODO: sort
        # embed input
        src_embed = self.embed(src)  # x: [B, S, D]

        # feed input to encoder RNN
        packed = pack_padded_sequence(src_embed, src_lens, batch_first=True)
        encoder_hidden, encoder_final = self.encoder_rnn(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]

        # get encoder final state, will be used as decoder initial state
        fwd_final = encoder_final[0:encoder_final.size(0):2]
        bwd_final = encoder_final[1:encoder_final.size(0):2]
        encoder_final = torch.cat([fwd_final, bwd_final], dim=2)  # encoder_final: [num_layers, B, 2H]

        trg_embed = self.embed(trg)
        max_len = self.args.sum_max_len
        hidden = torch.tanh(self.init_hidden(encoder_final))
        context_hidden = hidden[-1].unsqueeze(1)  # context_hidden指融合了context信息的hidden，初始化为hidden[-1]

        # pre-compute projected encoder hidden states(the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            if i == 0:  # <SOS> embedding
                prev_embed = self.embed(torch.LongTensor([1]).cuda()).repeat(len(src), 1).unsqueeze(1)
            else:
                if not test:  # last trg word embedding
                    prev_embed = trg_embed[:, i - 1].unsqueeze(1)
                else:  # last predicted word embedding
                    prev_idx = torch.argmax(pre_output_vectors[-1], dim=-1)
                    prev_embed = self.embed(prev_idx)
            hidden, context_hidden, pre_output = self.decode_step(prev_embed, encoder_hidden, src_mask, proj_key,
                                                                  hidden, context_hidden)
            pre_output_vectors.append(F.log_softmax(pre_output, dim=-1))

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
        self.alphas = alphas  # [B, 1, max_len]

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)  # [B, 1, 2*H]

        # context shape: [B, 1, 2*H], alphas shape: [B, 1, max_len]
        return context, alphas


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

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
