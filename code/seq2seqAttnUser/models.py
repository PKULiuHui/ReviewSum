# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):

    def __init__(self, args, embed):
        super(EncoderDecoder, self).__init__()
        self.name = 'seq2seqAttnUser'
        self.args = args
        # Embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # User embedding
        self.user_embed = nn.Embedding(args.user_num, args.user_dim)
        # Encoder
        self.encoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers,
                                  batch_first=True, bidirectional=True, dropout=args.encoder_dropout)
        # Attention
        self.attention = BahdanauAttention(args.hidden_size, query_size=args.hidden_size + args.user_dim)
        # Decoder
        self.decoder_rnn = nn.GRU(args.embed_dim + 2 * args.hidden_size + args.user_dim, args.hidden_size, args.num_layers,
                                  batch_first=True, dropout=args.decoder_dropout)
        self.init_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)
        self.pre_output_layer = nn.Linear(args.hidden_size + 2 * args.hidden_size + args.embed_dim + args.user_dim, args.hidden_size)
        # Generator
        self.generator = nn.Linear(args.hidden_size, args.embed_num)

    def decode_step(self, prev_embed, user_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = torch.cat([hidden[-1].unsqueeze(1), user_embed], dim=2)  # [B, 1, H]
        context, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, user_embed, context], dim=2)
        output, hidden = self.decoder_rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, user_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, src, trg, src_mask, src_lengths, trg_lengths, src_user, test=False):
        # embed user
        user_embed = self.user_embed(src_user).unsqueeze(1)

        # embed input
        src_embed = self.embed(src)  # x: [B, S, D]

        # feed input to encoder RNN
        packed = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        encoder_hidden, encoder_final = self.encoder_rnn(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]

        # get encoder final state, will be used as decoder initial state
        fwd_final = encoder_final[0:encoder_final.size(0):2]
        bwd_final = encoder_final[1:encoder_final.size(0):2]
        encoder_final = torch.cat([fwd_final, bwd_final], dim=2)  # encoder_final: [num_layers, B, 2H]

        trg_embed = self.embed(trg)
        max_len = self.args.sum_max_len
        hidden = torch.tanh(self.init_hidden(encoder_final))

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
            output, hidden, pre_output = self.decode_step(prev_embed, user_embed, encoder_hidden, src_mask, proj_key, hidden)
            pre_output_vectors.append(F.log_softmax(self.generator(pre_output), dim=-1))

        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)

        return pre_output_vectors

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)


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
        self.alphas = alphas  # [B, 1, max_len]

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)  # [B, 1, 2*H]

        # context shape: [B, 1, 2*H], alphas shape: [B, 1, max_len]
        return context, alphas
