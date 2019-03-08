# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):

    def __init__(self, args, embed):
        super(EncoderDecoder, self).__init__()
        self.name = 'HSSC'
        self.args = args
        # Embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # Encoder
        self.encoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers,
                                  batch_first=True, bidirectional=True, dropout=args.encoder_dropout)
        # Attention
        self.sum_attention = Attention(args.hidden_size)
        self.rating_attention = Attention(args.hidden_size)
        # Decoder
        self.decoder_rnn = nn.GRU(args.embed_dim + args.hidden_size, args.hidden_size, args.num_layers,
                                  batch_first=True, dropout=args.decoder_dropout)
        self.context_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)
        self.generator = nn.Linear(args.hidden_size, args.embed_num)
        self.fc = nn.Linear(args.hidden_size, args.hidden_size)
        self.classifier = nn.Linear(args.hidden_size, args.rating_range)

    def decode_step(self, prev_embed, encoder_hidden, src_mask, sum_key, rating_key, hidden,
                    context_hidden):
        """Perform a single decoder step (1 word)"""

        # update rnn hidden state
        # print(prev_embed.size())
        rnn_input = torch.cat([prev_embed, context_hidden], dim=-1)
        output, hidden = self.decoder_rnn(rnn_input, hidden)
        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)
        sum_context, _ = self.sum_attention(query=query, proj_key=sum_key, value=encoder_hidden, mask=src_mask)
        rating_context, _ = self.rating_attention(query=query, proj_key=rating_key, value=encoder_hidden, mask=src_mask)

        context_hidden = F.tanh(self.context_hidden(torch.cat([query, sum_context], dim=-1)))
        rating_context = rating_context.squeeze(1)

        pre_output = self.dropout_layer(context_hidden)
        pre_output = self.generator(pre_output)

        return hidden, pre_output, rating_context, context_hidden

    def forward(self, src, trg, src_mask, src_lengths, trg_lengths, test=False):
        # embed input
        src_embed = self.embed(src)  # x: [B, S, D]

        # feed input to encoder RNN
        packed = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        encoder_hidden, encoder_final = self.encoder_rnn(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]
        encoder_hidden = encoder_hidden[:, :, :self.args.hidden_size] + encoder_hidden[:, :, self.args.hidden_size:]

        fwd_final = encoder_final[0:encoder_final[0].size(0):2]
        bwd_final = encoder_final[1:encoder_final[0].size(0):2]
        hidden = fwd_final + bwd_final  # [num_layers, B, H]
        """
        # get encoder final state, will be used as decoder initial state
        fwd_final = encoder_final[0][0:encoder_final[0].size(0):2]
        bwd_final = encoder_final[0][1:encoder_final[0].size(0):2]
        hidden = fwd_final + bwd_final  # [num_layers, B, H]
        fwd_final = encoder_final[1][0:encoder_final[1].size(0):2]
        bwd_final = encoder_final[1][1:encoder_final[1].size(0):2]
        cell_state = fwd_final + bwd_final  # [num_layers, B, H]
        """

        trg_embed = self.embed(trg)
        max_len = self.args.sum_max_len

        sum_key = self.sum_attention.key_layer(encoder_hidden)
        rating_key = self.rating_attention.key_layer(encoder_hidden)
        context_hidden = hidden[-1].unsqueeze(1)
        pre_output_vectors = []
        rating_contexts = []
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
            hidden, pre_output, rating_context, context_hidden = self.decode_step(prev_embed,
                                                                                  encoder_hidden, src_mask,
                                                                                  sum_key, rating_key,
                                                                                  hidden,
                                                                                  context_hidden)
            pre_output_vectors.append(F.log_softmax(pre_output, dim=-1))
            rating_contexts.append(rating_context)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)

        rating_contexts = torch.cat(rating_contexts, dim=1).view(len(src), -1, self.args.hidden_size)
        classifier_input = torch.cat([rating_contexts, encoder_hidden], dim=1)
        classifier_input, _ = torch.max(classifier_input, dim=1)
        rating_output = F.log_softmax(self.classifier(F.relu(self.fc(classifier_input))), dim=-1)

        return pre_output_vectors, rating_output

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)


class Attention(nn.Module):

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(Attention, self).__init__()
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = hidden_size if key_size is None else key_size
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
