# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):

    def __init__(self, args, encoder, decoder, embed, generator):
        super(EncoderDecoder, self).__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        self.generator = generator

    def forward(self, src, trg, src_mask, src_lengths, trg_lengths, test=False):  # 测试的时候不能用标准答案
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encoder(self.embed(src), src_lengths)
        _1, _2, pre_output = self.decoder(self.embed(trg), encoder_hidden, encoder_final, src_mask, trg_lengths, test)
        return self.generator(pre_output)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, args):
        super(Generator, self).__init__()
        self.proj = nn.Linear(args.hidden_size, args.embed_num, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers,
                          batch_first=True, bidirectional=True, dropout=args.encoder_dropout)

    def forward(self, x, lengths):  # x: batch * max_len * D
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # output: batch * max_len * (2*H)
        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # final: num_layers * batch * (2*H)
        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, args, attention):
        super(Decoder, self).__init__()
        self.args = args
        self.attention = attention

        self.rnn = nn.GRU(args.embed_dim + 2 * args.hidden_size, args.hidden_size, args.num_layers,
                          batch_first=True, dropout=args.decoder_dropout)

        # to initialize from the final encoder state
        self.init_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)
        self.pre_output_layer = nn.Linear(args.hidden_size + 2 * args.hidden_size + args.embed_dim, args.hidden_size)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [B, 1, H]
        context, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final, src_mask, trg_lens, test):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        max_len = self.args.sum_max_len

        # initialize decoder hidden state
        hidden = torch.tanh(self.init_hidden(encoder_final))

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]


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
