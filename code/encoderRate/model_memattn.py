# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRating(nn.Module):

    def __init__(self, args, embed=None):
        super(EncoderRating, self).__init__()
        self.name = 'encoderRate + memory + user/product attention model'
        self.args = args

        # Word embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # Text Encoder
        self.review_encoder = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers, batch_first=True,
                                     bidirectional=True, dropout=args.dropout)
        # Summary Encoder
        self.sum_encoder = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers, batch_first=True,
                                  bidirectional=True, dropout=args.dropout)
        # Memory Scoring layer
        self.review_sim_1 = nn.Linear(2 * args.hidden_size, 1)
        self.review_sim_2 = nn.Linear(2 * args.hidden_size, 1)
        self.new_query = nn.Linear(4 * args.hidden_size, 2 * args.hidden_size)  # [query:mem_out] => new_query
        # Memory fusion layer
        self.mem_fusion = nn.Linear(4 * args.hidden_size, 2 * args.hidden_size)  # [u_mem_out:p_mem_out] => mem_out

        # Rating score prediction
        self.u_attn = Attention(args.hidden_size, key_size=2 * args.hidden_size, query_size=args.attr_dim)
        self.p_attn = Attention(args.hidden_size, key_size=2 * args.hidden_size, query_size=args.attr_dim)
        self.rat_predict_1 = nn.Linear(6 * args.hidden_size + 2 * args.attr_dim, args.hidden_size)
        self.rat_predict_2 = nn.Linear(args.hidden_size, args.rating_num)

        # User embedding layer
        self.user_embed = nn.Embedding(args.user_num, args.attr_dim)
        # Product embedding layer
        self.product_embed = nn.Embedding(args.product_num, args.attr_dim)

    def forward(self, src, trg, src_, trg_, user, product, rating, vocab_size, u_review, u_sum, p_review, p_sum):
        # useful variables
        batch_size = len(src)
        mem_size = self.args.mem_size
        src_lens = torch.sum(torch.sign(src), dim=1).tolist()
        src_mask = torch.sign(src).data
        u_review_lens = torch.sum(torch.sign(u_review), dim=1).data
        u_sum_lens = torch.sum(torch.sign(u_sum), dim=1).data
        p_review_lens = torch.sum(torch.sign(p_review), dim=1).data
        p_sum_lens = torch.sum(torch.sign(p_sum), dim=1).data

        # sort mem_review and mem_sum according to text lens, save idxs
        u_review_lens, u_review_idx = torch.sort(u_review_lens, descending=True)
        u_review = torch.index_select(u_review, 0, u_review_idx)
        u_sum_lens, u_sum_idx = torch.sort(u_sum_lens, descending=True)
        u_sum = torch.index_select(u_sum, 0, u_sum_idx)
        p_review_lens, p_review_idx = torch.sort(p_review_lens, descending=True)
        p_review = torch.index_select(p_review, 0, p_review_idx)
        p_sum_lens, p_sum_idx = torch.sort(p_sum_lens, descending=True)
        p_sum = torch.index_select(p_sum, 0, p_sum_idx)

        # embed text(src, mem_review, mem_sum)
        src_ = self.embed(src_)  # x: [B, S, D]
        u_review = self.embed(u_review)
        u_sum = self.embed(u_sum)
        p_review = self.embed(p_review)
        p_sum = self.embed(p_sum)

        # feed text into encoders
        packed = pack_padded_sequence(src_, src_lens, batch_first=True)
        encoder_hidden, encoder_final = self.review_encoder(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]
        fwd_final = encoder_final[0:encoder_final.size(0):2]
        bwd_final = encoder_final[1:encoder_final.size(0):2]
        encoder_final = torch.cat([fwd_final, bwd_final], dim=2).transpose(0, 1)  # encoder_final: [B, num_layers, 2H]

        packed = pack_padded_sequence(u_review, u_review_lens, batch_first=True)
        _, review_final = self.review_encoder(packed)
        u_review_final = \
            torch.cat([review_final[0:review_final.size(0):2], review_final[1:review_final.size(0):2]], dim=2)[-1]
        _, idx2 = torch.sort(u_review_idx)
        u_review_final = torch.index_select(u_review_final, 0, idx2)

        packed = pack_padded_sequence(p_review, p_review_lens, batch_first=True)
        _, review_final = self.review_encoder(packed)
        p_review_final = \
            torch.cat([review_final[0:review_final.size(0):2], review_final[1:review_final.size(0):2]], dim=2)[-1]
        _, idx2 = torch.sort(p_review_idx)
        p_review_final = torch.index_select(p_review_final, 0, idx2)

        packed = pack_padded_sequence(u_sum, u_sum_lens, batch_first=True)
        _, sum_final = self.sum_encoder(packed)
        u_sum_final = torch.cat([sum_final[0:sum_final.size(0):2], sum_final[1:sum_final.size(0):2]], dim=2)[-1]
        _, idx2 = torch.sort(u_sum_idx)
        u_sum_final = torch.index_select(u_sum_final, 0, idx2)

        packed = pack_padded_sequence(p_sum, p_sum_lens, batch_first=True)
        _, sum_final = self.sum_encoder(packed)
        p_sum_final = torch.cat([sum_final[0:sum_final.size(0):2], sum_final[1:sum_final.size(0):2]], dim=2)[-1]
        _, idx2 = torch.sort(p_sum_idx)
        p_sum_final = torch.index_select(p_sum_final, 0, idx2)

        u_query, p_query = encoder_final[:, -1], encoder_final[:, -1]
        for i in range(self.args.mem_layers):
            review_sim_1 = self.review_sim_1(u_query).unsqueeze(1)
            review_sim_2 = self.review_sim_2(u_review_final.view(batch_size, mem_size, -1))
            key_score = F.softmax((review_sim_1 + review_sim_2).view(batch_size, mem_size), dim=-1)
            u_mem_out = torch.bmm(key_score.view(batch_size, 1, mem_size),
                                  u_sum_final.view(batch_size, mem_size, -1)).view(batch_size, -1)
            u_query = self.new_query(torch.cat([u_query, u_mem_out], dim=-1))

            review_sim_1 = self.review_sim_1(p_query).unsqueeze(1)
            review_sim_2 = self.review_sim_2(p_review_final.view(batch_size, mem_size, -1))
            key_score = F.softmax((review_sim_1 + review_sim_2).view(batch_size, mem_size), dim=-1)
            p_mem_out = torch.bmm(key_score.view(batch_size, 1, mem_size),
                                  p_sum_final.view(batch_size, mem_size, -1)).view(batch_size, -1)
            p_query = self.new_query(torch.cat([p_query, p_mem_out], dim=-1))
        mem_out = self.mem_fusion(torch.cat([u_mem_out, p_mem_out], dim=-1)).unsqueeze(1)  # mem_out: [B, 1, 2H]

        user_embed = self.user_embed(user)  # user_embed: [B, A]
        product_embed = self.product_embed(product)  # product_embed: [B, A]

        # predict rating score
        proj_key_u = self.u_attn.key_layer(encoder_hidden)
        c1, _ = self.u_attn(query=user_embed.unsqueeze(1), proj_key=proj_key_u, value=encoder_hidden, mask=src_mask)
        proj_key_p = self.p_attn.key_layer(encoder_hidden)
        c2, _ = self.p_attn(query=product_embed.unsqueeze(1), proj_key=proj_key_p, value=encoder_hidden, mask=src_mask)
        rat_input = torch.cat([user_embed, product_embed, c1.squeeze(1), c2.squeeze(1), mem_out.squeeze(1)], dim=-1)
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
