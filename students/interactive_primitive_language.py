import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import models
from .primitive_language import PrimitiveLanguageStudent


class InteractivePrimitiveLanguageStudent(PrimitiveLanguageStudent):

    def prepare(self, world):
        self.world = world

    def set_tasks(self, tasks, is_eval):

        if is_eval:
            self.main_model.eval()
        else:
            self.main_model.train()

        # Init main model
        task_encodings = [task.encoding for task in tasks]
        #task_encodings = [list(reversed(task.encoding)) for task in tasks]
        task_encodings = self._to_tensor(task_encodings).long()
        self.main_model.init(self.batch_size, task_encodings)

        self.main_time = 0

    def set_instructions(self, instructions, is_eval):

        if is_eval:
            self.instructed_model.eval()
        else:
            self.instructed_model.train()

        # Init instructed model
        instruction_encodings, instruction_mask = \
            self._encode_and_pad(instructions)
        self.instructed_model.init(self.batch_size,
            instruction_encodings, src_mask=instruction_mask)

        self.local_state_seqs = []
        self.local_action_seqs = []

        self.instructed_time = 0

    def terminate(self, i):
        self.has_terminated[i] = True

    def init(self, states):
        self.batch_size = len(states)

        self.main_action_logit_seqs = []
        self.main_ref_action_seqs = []

        self.instructed_action_logit_seqs = []
        self.instructed_ref_action_seqs = []

        self.global_state_seqs = []
        self.global_action_seqs = []

        self.has_terminated = [False] * self.batch_size

    def receive(self, descriptions):

        state_seqs = self.local_state_seqs[:]
        action_seqs = self.local_action_seqs[:]

        self.set_instructions(descriptions, is_eval=False)

        zipped_info = zip(state_seqs, action_seqs)
        for t, (state_features, actions) in enumerate(zipped_info):
            time_feature = torch.tensor([t] * self.batch_size).to(self.device)
            action_logits = self.instructed_model.decode(state_features, time_feature)
            self.instructed_action_logit_seqs.append(action_logits)
            self.instructed_ref_action_seqs.append(actions)

    def imitate_instructed(self):
        zipped_info = zip(self.global_state_seqs, self.global_action_seqs)
        for t, (state_features, actions) in enumerate(zipped_info):
            time_feature = torch.tensor([t] * self.batch_size).to(self.device)
            action_logits = self.main_model.decode(state_features, time_feature)
            self.main_action_logit_seqs.append(action_logits)
            self.main_ref_action_seqs.append(actions)

    def act(self, states):
        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        time_feature = torch.tensor([self.main_time] * self.batch_size).to(self.device)
        action_logits = self.main_model.decode(state_features, time_feature)
        if self.main_model.training:
            actions = D.Categorical(logits=action_logits).sample()
        else:
            actions = action_logits.max(dim=1)[1]

        self.main_time += 1

        return actions.tolist()

    def instructed_act(self, states):
        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        time_feature = torch.tensor([self.instructed_time] * self.batch_size).to(self.device)
        action_logits = self.instructed_model.decode(state_features, time_feature)
        if self.instructed_model.training:
            actions = D.Categorical(logits=action_logits).sample()
        else:
            actions = action_logits.max(dim=1)[1]

        for i in range(self.batch_size):
            if self.has_terminated[i]:
                actions[i] = -1

        self.local_action_seqs.append(actions)
        self.local_state_seqs.append(state_features)

        self.global_action_seqs.append(actions)
        self.global_state_seqs.append(state_features)

        self.instructed_time += 1

        return actions.tolist()

    def _compute_loss(self, logit_seqs, target_seqs):
        assert len(logit_seqs) == len(target_seqs)

        loss = 0
        for logits, targets in zip(logit_seqs, target_seqs):
            loss += self.loss_fn(logits, targets)

        return loss

    def learn(self):

        instructed_loss = self._compute_loss(self.instructed_action_logit_seqs,
            self.instructed_ref_action_seqs)
        main_loss = self._compute_loss(self.main_action_logit_seqs,
            self.main_ref_action_seqs)

        loss = instructed_loss + main_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return instructed_loss.item() / len(self.instructed_action_logit_seqs) #+ \
        #main_loss.item() / len(self.main_action_logit_seqs)

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = {
            'instructed_model_state_dict': self.instructed_model.state_dict(),
            'main_model_state_dict': self.main_model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        self.instructed_model.load_state_dict(ckpt['instructed_model_state_dict'])
        self.main_model.load_state_dict(ckpt['main_model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





