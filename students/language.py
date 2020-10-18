import os
import sys
import json
import logging
import random
import re

import torch
import torch.nn as nn
import torch.distributions as D


from .imitation import ImitationStudent
from models.beam import Beam

import models


class LanguageStudent(ImitationStudent):

    def __init__(self, config):

        self.config = config
        self.device = config.device

        self.task_vocab = config.word_vocab
        self.src_vocab = config.char_vocab
        self.tgt_vocab = config.regex_vocab

        model_config = self.config.student.model
        model_config.device = config.device
        model_config.task_vocab_size = len(self.task_vocab)
        model_config.src_vocab_size = len(self.src_vocab)
        model_config.tgt_vocab_size = len(self.tgt_vocab)

        model_config.task_pad_idx = self.task_vocab['<PAD>']
        model_config.src_pad_idx = self.src_vocab['<PAD>']
        model_config.tgt_pad_idx = self.tgt_vocab['<PAD>']
        model_config.n_actions = len(self.tgt_vocab)

        self.exploration_model = models.load(model_config).to(self.device)
        self.execution_model   = models.load(model_config).to(self.device)

        logging.info('exploration model: ' + str(self.exploration_model))
        logging.info('execution model: '   + str(self.execution_model))

        self.optim = torch.optim.Adam(
            list(self.exploration_model.parameters()) +
            list(self.execution_model.parameters()),
            lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.tgt_pad_idx)

        self.random = random
        self.random.seed(123)

        with open('data/unlabeled_regexes.txt') as f:
            data = f.readlines()
            self.unlabeled_data = [list('<' + x.rstrip() + '>') for x in data]

        self.n_samples = self.config.student.n_samples

    def set_model(self, name):
        if name == 'exploration':
            self.model = self.exploration_model
        else:
            assert name == 'execution', name
            self.model = self.execution_model

    def reset(self):
        self.train_src_words = []
        self.train_tasks = []
        self.train_regexes = []

    def init(self, model_name, srcs, tasks, is_eval):
        self.set_model(model_name)
        super(LanguageStudent, self).init(srcs, tasks, is_eval)

    def receive(self, src_words, tasks, regexes):
        self.train_src_words.extend(src_words)
        self.train_tasks.extend(tasks)
        self.train_regexes.extend(regexes)

    """
    def beam_sample(self, src_words, tasks, n_samples=None):

        if n_samples is None:
            n_samples = self.n_samples

        batch_size = len(src_words)

        beam_src_words = []
        beam_tasks = []
        for src_word, task in zip(src_words, tasks):
            for _ in range(n_samples):
                beam_src_words.append(src_word)
                beam_tasks.append(task)

        self.init('exploration', beam_src_words, beam_tasks, True)

        beams = [Beam(n_samples, self.tgt_vocab, self.device, n_best=n_samples)
                    for _ in range(batch_size)]

        t = 0
        while self.timer > 0:

            t += 1

            prev_actions = torch.cat([b.get_last_token() for b in beams], dim=0)
            action_logits = self.model.decode(prev_actions)

            action_ldists = action_logits.log_softmax(dim=1)
            action_ldists = action_ldists.view(batch_size, n_samples, -1)

            prev_pos = []
            for i, (b, scores) in enumerate(zip(beams, action_ldists)):
                b.advance(scores, debug=i == 0 and t == 5)
                offset = n_samples * i
                prev_pos.append(offset + b.get_last_pointer())

            prev_pos = torch.cat(prev_pos, dim=0)
            self.model.index_select_decoder_state(prev_pos)

            self.timer -= 1
            if self.timer <= 0 or all([b.done() for b in beams]):
                break

        all_steps, all_last_positions = [], []

        pred_regexes = []
        for j, b in enumerate(beams):
            _, steps, last_positions = b.sort_finished(minimum=None)
            i = self.random.randint(0, min(len(steps), n_samples) - 1)
            seq = b.get_seq(steps[i], last_positions[i])
            pred_regexes.append(seq)

        return pred_regexes
    """

    def predict(self, src_words, tasks, sample=False):

        self.init('execution', src_words, tasks, True)
        while not self.has_terminated():
            self.act(sample=sample)

        return self.get_action_seqs()

    def compute_supervised_loss(self, model_name):

        batch_size = len(self.train_src_words)

        if batch_size == 0:
            return 0

        self.init(model_name, self.train_src_words, self.train_tasks, False)

        t = 0
        golds = [None] * batch_size
        while not self.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(self.train_regexes[i]):
                    golds[i] = self.train_regexes[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            self.act(gold_actions=golds)
            t += 1

        loss = super(LanguageStudent, self).compute_loss()

        return loss

    def compute_unsupervised_loss(self, model_name):

        batch_size = len(self.train_src_words)

        if batch_size == 0:
            return 0

        """
        regexes = []
        for i in range(batch_size):
            src_word = self.train_src_words[i][1:-1]
            while True:
                regex = self.random.choice(self.unlabeled_data)
                before, after = ''.join(regex)[1:-1].split('@')
                tgt_word = re.sub(before, after, src_word)
                if tgt_word != src_word:
                    regexes.append(regex)
                    break
        """
        regexes = self.random.sample(self.unlabeled_data, batch_size)

        self.init(model_name, self.train_src_words, self.train_tasks, False)

        t = 0
        golds = [None] * batch_size
        while not self.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(regexes[i]):
                    golds[i] = regexes[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            self.act(gold_actions=golds)
            t += 1

        loss = super(LanguageStudent, self).compute_loss()

        return loss

    def learn(self):

        assert len(self.train_src_words) == len(self.train_tasks) == len(self.train_regexes)

        exp_loss = self.compute_supervised_loss('exploration') + \
                   self.compute_unsupervised_loss('exploration')
        exe_loss = self.compute_supervised_loss('execution')

        loss = exp_loss + exe_loss

        if isinstance(loss, int) or isinstance(loss, float):
            return 0

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(self.gold_action_seqs)

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = { 'exp_model_state_dict': self.exploration_model.state_dict(),
                 'exe_model_state_dict': self.execution_model.state_dict(),
                 'optim_state_dict': self.optim.state_dict() }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            self.exploration_model.load_state_dict(ckpt['model_state_dict'])
            self.execution_model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.exploration_model.load_state_dict(ckpt['exp_model_state_dict'])
            self.execution_model.load_state_dict(ckpt['exe_model_state_dict'])
        #self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)
