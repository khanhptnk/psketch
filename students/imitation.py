import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import models


class ImitationStudent(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device

        model_config = config.student.model
        model_config.device = config.device
        model_config.vocab_size = len(config.vocab)
        model_config.pad_idx = config.vocab['<PAD>']
        model_config.enc_hidden_size = config.student.model.hidden_size
        model_config.dec_hidden_size = config.student.model.hidden_size

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=config.student.model.learning_rate)
        if hasattr(config.student.model, 'load_from'):
            self.load(config.student.model.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def prepare(self, world):
        self.world = world

    def init(self, tasks, states, is_eval):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval

        self.batch_size = len(states)

        self.ref_action_seqs = []
        self.action_logit_seqs = []
        self.has_terminated = [False] * self.batch_size

        self.time = 0

        task_encodings = []
        for task in tasks:
            task_encodings.append(list(reversed(task.encoding)))
        task_encodings = self._to_tensor(task_encodings).long()

        self.model.init(self.batch_size, task_encodings)

    def receive(self, ref_actions):
        ref_actions = self._to_tensor(ref_actions).long()
        self.ref_action_seqs.append(ref_actions)

    def act(self, states):
        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        time_feature = torch.tensor([self.time] * self.batch_size).to(self.device)
        action_logits = self.model.decode(state_features, time_feature)
        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            actions = action_logits.max(dim=1)[1]
        else:
            actions = D.Categorical(logits=action_logits).sample()

        return actions.tolist()

    def learn(self):

        assert len(self.ref_action_seqs) == len(self.action_logit_seqs)

        loss = 0
        for logits, targets in zip(self.action_logit_seqs, self.ref_action_seqs):
            loss += self.loss_fn(logits, targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(self.ref_action_seqs)

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = { 'model_state_dict': self.model.state_dict(),
                 'optim_state_dict': self.optim.state_dict() }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





