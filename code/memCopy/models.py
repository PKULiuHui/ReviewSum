# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MemCopy(nn.Module):

    def __init__(self, args, embed):
        super(MemCopy, self).__init__()
        self.name = 'seq2seqAttnCopy + memory'
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
        self.context_hidden = nn.Linear(3 * args.hidden_size, args.hidden_size, bias=False)
        # generate mode probability layer
        self.gen_p = nn.Linear(3 * args.hidden_size + args.embed_dim, 1)
        # generate mode layer, context_hidden => word distribution over fixed vocab, P(changeable vocab) = 0
        self.generator = nn.Linear(args.hidden_size, args.embed_num, bias=False)
        # copy mode layer, no learnable paras, attn_scores => word distribution over src vocab, P(other vocab) = 0

    def decode_step(self, src, prev_embed, encoder_hidden, src_mask, proj_key, hidden, context_hidden, mem_out,
                    vocab_size):
        """Perform a single decoder step (1 word)"""

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context_hidden, mem_out], dim=2)
        output, hidden = self.decoder_rnn(rnn_input, hidden)

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [B, 1, H]
        context, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)

        # 计算generate mode下的word distribution，非固定词表部分概率为0
        context_hidden = F.tanh(self.context_hidden(torch.cat([query, context], dim=2)))
        gen_prob = F.softmax(self.generator(context_hidden), dim=-1)
        if vocab_size > gen_prob.size(2):
            gen_prob = torch.cat(
                [gen_prob, torch.zeros(gen_prob.size(0), gen_prob.size(1), vocab_size - gen_prob.size(2)).cuda()],
                dim=-1)

        # 计算copy mode下的word distribution，非src中的词概率为0
        src = src.unsqueeze(1)
        copy_prob = torch.zeros(src.size(0), src.size(1), vocab_size).cuda().scatter_add(2, src, attn_probs)

        # 计算generate的概率p
        gen_p = F.sigmoid(self.gen_p(torch.cat([context, query, prev_embed], -1)))
        mix_prob = gen_p * gen_prob + (1 - gen_p) * copy_prob
        return hidden, context_hidden, mix_prob

    def forward(self, src, trg, src_, trg_, vocab_size, mem_up, mem_review, mem_sum, test=False):
        # useful variables
        batch_size = len(src)
        mem_size = self.args.mem_size
        src_lens = torch.sum(torch.sign(src), dim=1).tolist()
        src_mask = torch.sign(src).data
        mem_review_lens = torch.sum(torch.sign(mem_review), dim=1)
        mem_sum_lens = torch.sum(torch.sign(mem_sum), dim=1)
        mem_up = mem_up.view(batch_size, mem_size, 2)

        # sort mem_review and mem_sum according to text lens, save idxs
        mem_review_lens, mem_review_idx = torch.sort(mem_review_lens, descending=True)
        mem_review_sorted = torch.index_select(mem_review, 0, mem_review_idx)
        mem_sum_lens, mem_sum_idx = torch.sort(mem_sum_lens, descending=True)
        mem_sum_sorted = torch.index_select(mem_sum, 0, mem_sum_idx)

        # embed text(src, mem_review, mem_sum)
        src_embed = self.embed(src_)  # x: [B, S, D]
        mem_review_embed = self.embed(mem_review_sorted)
        mem_sum_embed = self.embed(mem_sum_sorted)

        # feed text into encoders
        packed = pack_padded_sequence(src_embed, src_lens, batch_first=True)
        encoder_hidden, encoder_final = self.review_encoder(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]
        fwd_final = encoder_final[0:encoder_final.size(0):2]
        bwd_final = encoder_final[1:encoder_final.size(0):2]
        encoder_final = torch.cat([fwd_final, bwd_final], dim=2)  # encoder_final: [num_layers, B, 2H]

        review_packed = pack_padded_sequence(mem_review_embed, mem_review_lens, batch_first=True)
        _, review_final = self.review_encoder(review_packed)
        review_final = \
            torch.cat([review_final[0:review_final.size(0):2], review_final[1:review_final.size(0):2]], dim=2)[-1]
        _, idx2 = torch.sort(mem_review_idx)
        review_final = torch.index_select(review_final, 0, idx2)

        sum_packed = pack_padded_sequence(mem_sum_embed, mem_sum_lens, batch_first=True)
        _, sum_final = self.sum_encoder(sum_packed)
        sum_final = torch.cat([sum_final[0:sum_final.size(0):2], sum_final[1:sum_final.size(0):2]], dim=2)[-1]
        _, idx2 = torch.sort(mem_sum_idx)
        sum_final = torch.index_select(sum_final, 0, idx2)

        query = encoder_final[-1]
        for i in range(self.args.mem_layers):
            query_extend = query.repeat(1, mem_size).view(-1, query.size(-1))
            review_sim = torch.bmm(query_extend.view(query_extend.size(0), 1, -1),
                                   review_final.view(review_final.size(0), -1, 1)).view(batch_size, mem_size, 1)
            key_score = F.softmax(self.key_score(torch.cat([mem_up, review_sim], dim=-1)).view(batch_size, mem_size),
                                  dim=-1)
            mem_out = torch.bmm(key_score.view(batch_size, 1, mem_size), sum_final.view(batch_size, mem_size, -1)).view(
                batch_size, -1)
            query = self.new_query(torch.cat([query, mem_out], dim=-1))

        trg_embed = self.embed(trg_)
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
                    for j in range(0, prev_idx.size(0)):
                        if prev_idx[j][0] >= self.args.embed_num:
                            prev_idx[j][0] = 3  # UNK_IDX
                    prev_embed = self.embed(prev_idx)
            hidden, context_hidden, word_prob = self.decode_step(src, prev_embed, encoder_hidden, src_mask, proj_key,
                                                                 hidden, context_hidden, mem_out.unsqueeze(1),
                                                                 vocab_size)
            pre_output_vectors.append(word_prob)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)

        return key_score, pre_output_vectors

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
    def __init__(self, args):
        super(MyLoss, self).__init__()
        assert args.loss_type == 'text' or args.loss_type == 'mix'
        self.type = args.loss_type
        self.temp = args.mem_loss_temp
        self.ratio = args.mem_loss_ratio

    def forward(self, mem_output, sum_output, mem_output_gold, sum_output_gold):
        nll_loss = nn.NLLLoss(ignore_index=0, size_average=False)
        loss_1 = nll_loss(sum_output, sum_output_gold)
        if self.type == 'text':
            return loss_1, loss_1, loss_1
        mem_output_gold = F.softmax(mem_output_gold * self.temp, dim=-1)
        kl_div_loss = nn.KLDivLoss(size_average=False)
        loss_2 = self.ratio * kl_div_loss(torch.log(mem_output + 1e-20), mem_output_gold)
        loss = loss_1 + loss_2
        return loss, loss_1, loss_2
