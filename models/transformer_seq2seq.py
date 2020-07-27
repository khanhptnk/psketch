import os
import sys
import math
import copy

import torch
import torch.nn as nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, timestep=None):
        if timestep is None:
            x = x + self.pe[:x.size(0), :]
        else:
            x = x + self.pe[timestep, :]
        return self.dropout(x)


class LayerNormResidual(nn.Module):

    def __init__(self, module, input_size, output_size, dropout):

        super(LayerNormResidual, self).__init__()

        self.module = module
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

        if input_size != output_size:
            self.shortcut_layer = nn.Linear(input_size, output_size)
        else:
            self.shortcut_layer = lambda x: x

    def forward(self, input, *args, **kwargs):
        input_shortcut = self.shortcut_layer(input)
        return self.layer_norm(
            input_shortcut + self.dropout(self.module(input, *args, **kwargs)))


class LearnedTime(nn.Module):

    def __init__(self, input_size, output_size):

        super(LearnedTime, self).__init__()
        module = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )
        self.increment_op = LayerNormResidual(
            module, input_size, output_size, 0)

        init_time = torch.zeros((output_size,), dtype=torch.float)
        self.register_buffer('init_time', init_time)

    def reset(self, batch_size):
        self.time = self.init_time.expand(batch_size, -1)
        return self.time

    def forward(self, other_time=None):
        if other_time is None:
            input_time = self.time
        else:
            input_time = torch.cat((self.time, other_time), dim=-1)
        self.time = self.increment_op(input_time)
        return self.time


class TransformerDecoderLayerWithSelfMemory(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayerWithSelfMemory, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead,
            dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead,
            dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, self_memory_key, self_memory_value, src_memory,
            src_memory_mask):

        tgt2 = self.self_attn(tgt, self_memory_key, self_memory_value)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, src_memory, src_memory,
            key_padding_mask=src_memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class EncoderTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
            dropout, num_layers, nhead, device):

        super(EncoderTransformer, self).__init__()

        #self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        #self.time_embedding = LearnedTime(hidden_size, hidden_size)
        self.time_embedding = nn.Embedding(100, embedding_size // 2)
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_size // 2, padding_idx)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.device = device

    def forward(self, input, mask=None):

        input = input.transpose(0, 1) # len x batch
        embed = self.embedding(input) * math.sqrt(input.size(-1))
        time_tensor = torch.tensor(
            [ [i] * input.size(1) for i in range(input.size(0)) ],
            device=self.device).long()
        time_embed = self.time_embedding(time_tensor)
        embed = torch.cat([embed, time_embed], dim=-1)

        context = self.transformer(embed, src_key_padding_mask=mask)

        return context


class DecoderTransformer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout, num_layers,
            nhead, device):

        super(DecoderTransformer, self).__init__()

        #self.time_embedding = LearnedTime(hidden_size, hidden_size)
        #self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.time_embedding = nn.Embedding(100, hidden_size // 2)

        self.embedding = nn.Linear(input_size, hidden_size // 2)
        self.drop = nn.Dropout(dropout)
        layer = TransformerDecoderLayerWithSelfMemory(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout)
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers

        init_h = torch.zeros((hidden_size,), dtype=torch.float)
        self.register_buffer('init_h', init_h)

        self.device = device

    def reset(self, batch_size, h0=None):
        #self.time = self.time_embedding.reset(batch_size)
        #self.time = 0
        self.keys = [[self.init_h.expand(batch_size, -1).contiguous()]
            for _ in range(self.num_layers)]
        if h0 is None:
            self.values = [[self.init_h.expand(batch_size, -1).contiguous()]
                for _ in range(self.num_layers)]
        else:
            self.values = [[h0] for _ in range(self.num_layers)]

    def forward(self, input, timestep, src_memory, src_memory_mask=None):

        time_embed = self.time_embedding(timestep)
        embed = self.embedding(input)
        input = self.drop(torch.cat([embed, time_embed], dim=-1))
        #embed = self.embedding(input)
        #self.time += 1
        #input = self.pos_encoder(embed, self.time)

        for i, layer in enumerate(self.layers):

            self_memory_key = torch.stack(self.keys[i])
            self_memory_value = torch.stack(self.values[i])

            output = layer(input.unsqueeze(0), self_memory_key,
                self_memory_value, src_memory, src_memory_mask).squeeze(0)

            self.keys[i].append(output)
            self.values[i].append(output)
            input = output

        return output


class TransformerSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(TransformerSeq2SeqModel, self).__init__()

        self.encoder = EncoderTransformer(
            config.vocab_size,
            config.word_embed_size,
            config.enc_hidden_size,
            config.pad_idx,
            config.dropout_ratio,
            config.num_layers,
            config.nhead,
            config.device)

        self.decoder = DecoderTransformer(
            config.input_size,
            config.dec_hidden_size,
            config.dropout_ratio,
            config.num_layers,
            config.nhead,
            config.device)

        self.predictor = nn.Linear(config.dec_hidden_size, config.n_actions)

        self.device = config.device

    def init(self, batch_size, src, src_mask=None):
        self.src = self.encoder(src)
        self.src_mask = src_mask
        self.decoder.reset(batch_size, h0=self.src[0])

    def decode(self, input, timestep):
        output = self.decoder(input, timestep, self.src, self.src_mask)
        logit = self.predictor(output)

        #print(logit[0].softmax(0).tolist())

        return logit

