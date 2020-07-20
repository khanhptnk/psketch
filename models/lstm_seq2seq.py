import os
import sys

import torch
import torch.nn as nn


from .base import LSTMWrapper, Attention


class EncoderLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
            dropout_ratio, device):

        super(EncoderLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = LSTMWrapper(
            embedding_size,
            hidden_size,
            dropout_ratio,
            device,
            batch_first=True)
        self.device = device

    def forward(self, input):

        embed = self.embedding(input)
        embed = self.drop(embed)

        h0 = self.lstm.init_state(input.size(0))
        context, (last_h, last_c) = self.lstm(embed, h0)

        return context, last_h, last_c


class DecoderLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_ratio, device):

        super(DecoderLSTM, self).__init__()

        self.drop = nn.Dropout(p=dropout_ratio)

        self.lstm = LSTMWrapper(
            input_size,
            hidden_size,
            dropout_ratio,
            device)
        self.device = device

    def init_state(self, batch_size, h0=None):
        #self.zeros_vec = torch.zeros(
        #    (batch_size, 1), dtype=torch.uint8, device=self.device)
        if h0 is None:
            self.h = self.lstm.init_state(batch_size)
        else:
            self.h = h0

    def forward(self, input):
        input_drop = self.drop(input)
        output, self.h = self.lstm(input_drop.unsqueeze(0), self.h)
        output = self.drop(output.squeeze(0))

        return output


class LSTMSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(LSTMSeq2SeqModel, self).__init__()

        self.encoder = EncoderLSTM(
            config.vocab_size,
            config.word_embed_size,
            config.enc_hidden_size,
            config.pad_idx,
            config.dropout_ratio,
            config.device)

        self.decoder = DecoderLSTM(
            config.input_size,
            config.dec_hidden_size,
            config.dropout_ratio,
            config.device)

        self.attention = Attention(
            config.hidden_size, config.hidden_size, config.hidden_size // 2)

        self.predictor = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.n_actions)
            )

        self.enc2dec = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )

        self.device = config.device

    def init(self, batch_size, seq, attention_mask=None):
        self.context, last_enc_h, last_enc_c = self.encoder(seq)

        last_enc_h = self.enc2dec(last_enc_h)

        self.decoder.init_state(batch_size, h0=(last_enc_h, last_enc_c))
        self.attention_mask = attention_mask

    def decode(self, obs, attention_mask=None):

        dec_output = self.decoder(obs)
        attended_context, _ = self.attention(
            dec_output, self.context, mask=self.attention_mask)

        feature = torch.cat([dec_output, attended_context], dim=1)
        logit = self.predictor(feature)

        return logit

    def encode(self, seq):
        return self.encoder(seq)
