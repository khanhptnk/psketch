import os
import sys

import torch
import torch.nn as nn


from .base import LSTMWrapper, Attention


class EncoderLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, padding_idx, dropout_ratio,
            device):

        super(EncoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = LSTMWrapper(
            input_size,
            hidden_size,
            dropout_ratio,
            device,
            batch_first=True)
        self.device = device

    def forward(self, input):

        h0 = self.lstm.init_state(input.size(0))
        context, (last_h, last_c) = self.lstm(input, h0)

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

    """
    def init_state(self, h0, reset_indices=None):
        if reset_indices is None:
            self.h = h0
        else:
            for idx in reset_indices:
                self.h[0][:, idx, :] = h0[0][:, idx, :]
                self.h[1][:, idx, :] = h0[1][:, idx, :]
    """

    def forward(self, input, h):
        input_drop = self.drop(input)
        output, new_h = self.lstm(input_drop.unsqueeze(0), h)
        output = self.drop(output.squeeze(0))

        return output, new_h


class LSTMSeq2SeqModel(nn.Module):

    def __init__(self, config):

        super(LSTMSeq2SeqModel, self).__init__()

        time_embed_size = 64

        self.encoder = EncoderLSTM(
            config.word_embed_size + time_embed_size,
            config.enc_hidden_size,
            config.pad_idx,
            config.dropout_ratio,
            config.device)

        dec_input_size = config.input_size
        if not config.no_time:
            dec_input_size += time_embed_size

        self.decoder = DecoderLSTM(
            dec_input_size,
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

        self.embedding = nn.Embedding(
                config.vocab_size,
                config.word_embed_size,
                config.pad_idx)
        self.src_time_embedding = nn.Embedding(100, time_embed_size)

        self.no_time = config.no_time
        if not self.no_time:
            self.tgt_time_embedding = nn.Embedding(100, time_embed_size)

        self.device = config.device
        self.n_actions = config.n_actions

    """
    def init(self, src, src_mask=None, reset_indices=None):

        batch_size = src.shape[0]

        time_tensor = torch.tensor([ list(range(src.size(1)))
            for _ in range(batch_size) ]).to(self.device)
        time_embed = self.src_time_embedding(time_tensor)
        src_embed = self.embedding(src)
        src_input = torch.cat([src_embed, time_embed], dim=2)
        self.context, last_enc_h, last_enc_c = self.encoder(src_input)
        self.src_mask = src_mask
        last_enc_h = self.enc2dec(last_enc_h)
        last_enc_state = (last_enc_h, last_enc_c)

        if reset_indices is None:
            self.time = torch.zeros(batch_size).long().to(self.device)
            self.dec_h = last_enc_state
        else:
            for idx in reset_indices:
                self.time[idx] = 0
                for i in range(2):
                    self.dec_h[i][:, idx, :] = last_enc_state[i][:, idx, :]
    """

    def encode(self, src, src_mask=None):

        batch_size = src.shape[0]

        time_tensor = torch.tensor([ list(range(src.size(1)))
            for _ in range(batch_size) ]).to(self.device)
        time_embed = self.src_time_embedding(time_tensor)
        src_embed = self.embedding(src)
        src_input = torch.cat([src_embed, time_embed], dim=2)
        self.context, last_enc_h, last_enc_c = self.encoder(src_input)
        self.src_mask = src_mask
        last_enc_h = self.enc2dec(last_enc_h)
        last_enc_state = (last_enc_h, last_enc_c)

        return last_enc_state

    def decode(self, obs, h, time):

        if not self.no_time:
            time_embed = self.tgt_time_embedding(time)
            input = torch.cat([obs, time_embed], dim=1)
        else:
            input = obs

        output, new_h = self.decoder(input, h)

        attended_context, _ = self.attention(
            output, self.context, mask=self.src_mask)

        feature = torch.cat([output, attended_context], dim=1)
        logit = self.predictor(feature)

        return logit, new_h, time + 1


