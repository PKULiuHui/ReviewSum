# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class EncoderDecoder(nn.Module):

    def __init__(self, args, embed):
        super(EncoderDecoder, self).__init__()
        self.name = 'seq2seq'
        self.args = args
        # Embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # Encoder
        self.encoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers,
                                  batch_first=True, bidirectional=True, dropout=args.encoder_dropout)
        # Decoder
        self.decoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers,
                                  batch_first=True, dropout=args.decoder_dropout)
        self.init_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)
        self.pre_output_layer = nn.Linear(args.hidden_size + args.embed_dim, args.hidden_size)
        # Generator
        self.generator = nn.Linear(args.hidden_size, args.embed_num)

    def decode_step(self, prev_embed, hidden):
        """Perform a single decoder step (1 word)"""
        output, hidden = self.decoder_rnn(prev_embed, hidden)
        pre_output = torch.cat([prev_embed, output], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, src, trg, src_mask, src_lengths, trg_lengths, test=False):
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

        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        teacher = True if np.random.random() < self.args.teacher else False
        for i in range(max_len):
            if i == 0:  # <SOS> embedding
                prev_embed = self.embed(torch.LongTensor([1]).cuda()).repeat(len(src), 1).unsqueeze(1)
            else:
                if not test and teacher:  # last trg word embedding
                    prev_embed = trg_embed[:, i - 1].unsqueeze(1)
                else:  # last predicted word embedding
                    prev_idx = torch.argmax(pre_output_vectors[-1], dim=-1)
                    prev_embed = self.embed(prev_idx)
            output, hidden, pre_output = self.decode_step(prev_embed, hidden)
            pre_output_vectors.append(F.log_softmax(self.generator(pre_output), dim=-1))

        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)

        return pre_output_vectors

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)
