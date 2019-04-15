# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):

    def __init__(self, args, embed=None):
        super(EncoderDecoder, self).__init__()
        self.name = 'seq2seqAttrRate2'
        self.args = args

        # Word embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)
        # Text Encoder
        self.encoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers, batch_first=True,
                                  bidirectional=True, dropout=args.encoder_dropout)
        self.text_final = nn.Linear(2 * args.hidden_size, args.hidden_size)

        # User embedding layer
        self.user_embed = nn.Embedding(args.user_num, args.attr_dim)
        # Product embedding layer
        self.product_embed = nn.Embedding(args.product_num, args.attr_dim)
        # Rating embedding layer
        self.rating_embed = nn.Embedding(args.rating_num, args.attr_dim)
        # Encoder final layer
        self.attr_final = nn.Linear(3 * args.attr_dim, args.hidden_size)
        # encoder gate
        self.encoder_gate = nn.Linear(2 * args.hidden_size + 3 * args.attr_dim, 1)

        # Highway
        self.highway_fusion = nn.Linear(3 * args.attr_dim, args.hidden_size)
        # Decoder
        self.decoder_rnn = nn.GRU(args.embed_dim + 2 * args.hidden_size, args.hidden_size, args.num_layers,
                                  batch_first=True, dropout=args.decoder_dropout)
        # Text Attention
        self.attention = Attention(args.hidden_size)
        self.text_context = nn.Linear(2 * args.hidden_size, args.hidden_size)
        # Attributes Attention
        self.attention_attr = Attention(args.hidden_size, key_size=args.attr_dim, query_size=args.hidden_size)
        self.attr_context = nn.Linear(args.attr_dim, args.hidden_size)
        # context gate
        self.context_gate = nn.Linear(3 * args.hidden_size + args.embed_dim + args.attr_dim, 1)

        # mix hidden and context into a context_hidden vector
        self.context_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size)

        # generate mode probability layer
        self.gen_p = nn.Linear(2 * args.hidden_size + args.embed_dim, 1)
        # Dropout layer before generator
        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)
        # generate mode layer, context_hidden => word distribution over fixed vocab, P(changeable vocab) = 0
        self.generator = nn.Linear(args.hidden_size, args.embed_num, bias=False)
        # copy mode layer, no learnable paras, attn_scores => word distribution over src vocab, P(other vocab) = 0

    def decode_step(self, src, prev_embed, encoder_hidden, src_mask, proj_key, encoder_attr,
                    proj_key_attr, hidden, context_hidden, vocab_size, highway):
        """Perform a single decoder step (1 word)"""

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context_hidden, highway], dim=2)
        output, hidden = self.decoder_rnn(rnn_input, hidden)

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [B, 1, H]
        context_text, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)
        context_attr, _ = self.attention_attr(query=query, proj_key=proj_key_attr, value=encoder_attr)

        text_context = self.text_context(context_text)
        attr_context = self.attr_context(context_attr)
        context_g = F.sigmoid(self.context_gate(torch.cat([query, prev_embed, context_attr, context_text], dim=-1)))
        context = context_g * text_context + (1 - context_g) * attr_context

        # 计算generate mode下的word distribution，非固定词表部分概率为0
        context_hidden = F.tanh(self.context_hidden(torch.cat([query, context], dim=2)))
        context_hidden_1 = self.dropout_layer(context_hidden)
        gen_prob = F.softmax(self.generator(context_hidden_1), dim=-1)
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

    # src_和src的区别在于对于不在固定词典中的词，src中序号为它在可变词典中的序号，src_中的序号为UNK_IDX，
    # 这样设置是为了方便embedding层，否则还要挨个判断每个词是否在固定词典中
    def forward(self, src, trg, src_, trg_, user, product, rating, vocab_size, test=False):
        # useful variables
        batch_size = len(src)
        src_lens = torch.sum(torch.sign(src), dim=1).tolist()
        src_mask = torch.sign(src).data

        # embed src words
        src_ = self.embed(src_)

        # feed input to encoder RNN
        packed = pack_padded_sequence(src_, src_lens, batch_first=True)
        encoder_hidden, encoder_final = self.encoder_rnn(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]
        fwd_final = encoder_final[0:encoder_final.size(0):2]
        bwd_final = encoder_final[1:encoder_final.size(0):2]
        encoder_final = torch.cat([fwd_final, bwd_final], dim=2).transpose(0, 1)  # encoder_final: [B, num_layers, 2H]
        text_final = self.text_final(encoder_final)  # text_final: [B, num_layers, H]

        # embed attributes
        user_embed = self.user_embed(user)  # user_embed: [B, A]
        product_embed = self.product_embed(product)  # product_embed: [B, A]
        rating_embed = self.rating_embed(rating)
        attr_final = self.attr_final(torch.cat([user_embed, product_embed, rating_embed], dim=-1)).unsqueeze(1)
        attr_final = attr_final.repeat(1, text_final.size(1), 1)  # attr_final: [B, num_layers, H]
        encoder_attr = torch.cat([user_embed, product_embed, rating_embed], dim=-1).view(user_embed.size(0), 3, -1)

        encoder_g = F.sigmoid(
            self.encoder_gate(torch.cat([encoder_final[:, -1], user_embed, product_embed, rating_embed], dim=-1)))
        encoder_final = encoder_g * text_final.contiguous().view(batch_size, -1) + (
                1 - encoder_g) * attr_final.contiguous().view(batch_size, -1)
        encoder_final = encoder_final.view(batch_size, self.args.num_layers, -1).transpose(0, 1)  # [num_layers, B, H]

        trg_embed = self.embed(trg_)
        max_len = self.args.sum_max_len
        hidden = encoder_final.contiguous()
        context_hidden = hidden[-1].unsqueeze(1)  # context_hidden指融合了context信息的hidden，初始化为hidden[-1]
        highway = self.highway_fusion(torch.cat([user_embed, product_embed, rating_embed], dim=-1)).unsqueeze(
            1).contiguous()

        # pre-compute projected encoder hidden states(the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        proj_key_attr = self.attention_attr.key_layer(encoder_attr)
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
            hidden, context_hidden, word_prob = self.decode_step(src, prev_embed, encoder_hidden, src_mask,
                                                                 proj_key, encoder_attr, proj_key_attr, hidden,
                                                                 context_hidden, vocab_size, highway)
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
