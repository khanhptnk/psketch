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
        context, _ = self.lstm(embed, h0)

        return context


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

    def init_state(self, batch_size):
        #self.zeros_vec = torch.zeros(
        #    (batch_size, 1), dtype=torch.uint8, device=self.device)
        self.h = self.lstm.init_state(batch_size)

    def forward(self, input):
        input_drop = self.drop(input)
        output, self.h = self.lstm(input_drop.unsqueeze(0), self.h)
        output = self.drop(output.squeeze(0))

        return output


class LSTMSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(LSTMSeq2SeqModel, self).__init__()

        hparams = config.student.model
        self.device = config.device

        enc_hidden_size = hparams.hidden_size

        self.encoder = EncoderLSTM(
            hparams.vocab_size,
            hparams.word_embed_size,
            enc_hidden_size,
            hparams.pad_idx,
            hparams.dropout_ratio,
            self.device)

        self.decoder = DecoderLSTM(
            hparams.input_size,
            hparams.hidden_size,
            hparams.dropout_ratio,
            self.device)

        self.attention = Attention(
            hparams.hidden_size, hparams.hidden_size, hparams.hidden_size // 2)

        self.predictor = nn.Sequential(
                nn.Linear(hparams.hidden_size * 2, hparams.hidden_size),
                nn.Tanh(),
                nn.Linear(hparams.hidden_size, hparams.n_actions)
            )

    def init(self, batch_size, seq):
        self.context = self.encoder(seq)
        self.decoder.init_state(batch_size)

    def decode(self, obs):

        dec_output = self.decoder(obs)
        attended_context, _ = self.attention(dec_output, self.context)

        feature = torch.cat([dec_output, attended_context], dim=1)
        logit = self.predictor(feature)

        return logit

    def encode(self, seq):
        return self.encoder(seq)
