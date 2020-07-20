import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import models
from .imitation import ImitationStudent


class PrimitiveLanguageStudent(ImitationStudent):

    def __init__(self, config):
        self.config = config
        self.vocab = config.vocab
        self.device = config.device

        model_config = config.student.model
        model_config.device = config.device
        model_config.vocab_size = len(config.vocab)
        model_config.pad_idx = config.vocab['<PAD>']
        model_config.enc_hidden_size = config.student.model.hidden_size
        model_config.dec_hidden_size = config.student.model.hidden_size

        self.instructed_model = models.load(model_config).to(self.device)
        self.main_model = models.load(model_config).to(self.device)

        logging.info('instructed_model: ' + str(self.instructed_model))
        logging.info('main_model: ' + str(self.main_model))

        lr = model_config.learning_rate
        self.instructed_optim = torch.optim.Adam(
            self.instructed_model.parameters(), lr=lr)
        self.main_optim = torch.optim.Adam(self.main_model.parameters(), lr=lr)

        if hasattr(config.student.model, 'load_from'):
            self.load(config.student.model.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _encode_and_pad(self, xs):
        # Init instructed model
        encodings = []
        masks = []
        for x in xs:
            encodings.append([self.vocab[w] for w in x])
            masks.append([0] * len(encodings[-1]))
        # Padding
        max_len = max([len(encoding) for encoding in encodings])

        for encoding in encodings:
            encoding.extend([self.vocab['<PAD>']] * (max_len - len(encoding)))

        for mask in masks:
            mask.extend([1] * (max_len - len(mask)))

        for i in range(self.batch_size):
            encodings[i] = list(reversed(encodings[i]))
            masks[i] = list(reversed(masks[i]))

        encodings = self._to_tensor(encodings).long()
        masks = self._to_tensor(masks).bool()

        return encodings, masks

    def prepare(self, world):
        self.world = world

    def set_tasks(self, tasks):
        # Init main model
        #task_encodings = [task.encoding for task in tasks]
        task_encodings = [list(reversed(task.encoding)) for task in tasks]
        task_encodings = self._to_tensor(task_encodings).long()
        self.main_model.init(self.batch_size, task_encodings)

    def set_instructions(self, instructions):
        # Init instructed model
        instruction_encodings, instruction_mask = \
            self._encode_and_pad(instructions)
        self.instructed_model.init(self.batch_size,
            instruction_encodings, attention_mask=instruction_mask)

    def reset_history(self):
        self.state_seqs = []
        self.action_seqs = []
        self.has_terminated = [False] * self.batch_size

    def init(self, tasks, instructions, states, is_eval):

        self.is_eval = is_eval
        self.batch_size = len(states)

        if is_eval:
            assert tasks is not None
            self.main_model.eval()
            self.set_tasks(tasks)
        else:
            self.main_model.train()
            self.set_tasks(tasks)
            self.instructed_model.train()
            self.set_instructions(instructions)

        self.reset_history()

    def receive(self, descriptions):
        description_encodings, description_mask = \
            self._encode_and_pad(descriptions)
        self.instructed_model.init(self.batch_size, description_encodings,
            attention_mask=description_mask)

        self.instructed_action_logit_seqs = []
        self.instructed_ref_action_seqs = []
        for state_features, actions in zip(self.state_seqs, self.action_seqs):
            action_logits = self.instructed_model.decode(state_features)
            self.instructed_action_logit_seqs.append(action_logits)
            self.instructed_ref_action_seqs.append(actions)

    def imitate_instructed(self):
        self.main_action_logit_seqs = []
        self.main_ref_action_seqs = []
        for state_features, actions in zip(self.state_seqs, self.action_seqs):
            action_logits = self.main_model.decode(state_features)
            self.main_action_logit_seqs.append(action_logits)
            self.main_ref_action_seqs.append(actions)

    def act(self, states):
        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        action_logits = self.main_model.decode(state_features)
        if self.main_model.training:
            actions = D.Categorical(logits=action_logits).sample()
        else:
            actions = action_logits.max(dim=1)[1]

        return actions.tolist()

    def instructed_act(self, states):
        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        action_logits = self.instructed_model.decode(state_features)
        if self.instructed_model.training:
            actions = D.Categorical(logits=action_logits).sample()
        else:
            actions = action_logits.max(dim=1)[1]

        for i in range(self.batch_size):
            terminate = actions[i].item() == self.world.actions.STOP.index
            if self.has_terminated[i]:
                actions[i] = -1
            self.has_terminated[i] |= terminate

        self.action_seqs.append(actions)
        self.state_seqs.append(state_features)

        return actions.tolist()

    def _learn_model(self, logit_seqs, target_seqs, optim):
        assert len(logit_seqs) == len(target_seqs)

        loss = 0
        for logits, targets in zip(logit_seqs, target_seqs):
            loss += self.loss_fn(logits, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item() / len(logit_seqs)

    def learn(self):

        instructed_loss = self._learn_model(self.instructed_action_logit_seqs,
            self.instructed_ref_action_seqs, self.instructed_optim)
        main_loss = self._learn_model(self.main_action_logit_seqs,
            self.main_ref_action_seqs, self.main_optim)

        return instructed_loss + main_loss

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = {
            'instructed_model_state_dict': self.instructed_model.state_dict(),
            'instructed_optim_state_dict': self.instructed_optim.state_dict(),
            'main_model_state_dict': self.main_model.state_dict(),
            'main_optim_state_dict': self.main_optim.state_dict()
        }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        self.instructed_model.load_state_dict(ckpt['instructed_model_state_dict'])
        self.instructed_optim.load_state_dict(ckpt['instructed_optim_state_dict'])
        self.main_model.load_state_dict(ckpt['main_model_state_dict'])
        self.main_optim.load_state_dict(ckpt['main_optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





