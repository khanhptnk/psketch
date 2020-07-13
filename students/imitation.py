import os
import sys
import json

import torch
import torch.nn as nn
import torch.distributions as D

import models


class ImitationStudent(object):

    def __init__(self, config, model_path=None):
        self.config = config
        self.device = config.device

        self.model = models.load(config).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=config.student.model.learning_rate)
        if model_path is not None:
            self.load(model_path)

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

        batch_size = len(states)

        self.ref_action_seqs = []
        self.action_logit_seqs = []
        self.has_terminated = [False] * batch_size

        task_encodings = []
        for task in tasks:
            task_encodings.append(task.encoding)
        task_encodings = self._to_tensor(task_encodings).long()

        self.model.init(batch_size, task_encodings)

    def receive(self, ref_actions):
        ref_actions = self._to_tensor(ref_actions).long()
        self.ref_action_seqs.append(ref_actions)

    def act(self, states):
        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        action_logits = self.model.decode(state_features)
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
        print('Saved %s model to %s' % (name, file_path))

        if trajectories is not None:
            file_path = os.path.join(self.config.experiment_dir, name + '.traj')
            with open(file_path, 'w') as f:
                json.dump(trajectories, f)
            print('Saved %s trajectories to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])





