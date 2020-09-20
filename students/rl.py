import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import models
from .imitation import ImitationStudent


class RLStudent(ImitationStudent):

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.task_vocab = config.word_vocab
        self.src_vocab = config.char_vocab
        self.tgt_vocab = config.regex_vocab

        model_config = config.student.model
        model_config.device = config.device
        model_config.task_vocab_size = len(self.task_vocab)
        model_config.src_vocab_size = len(self.src_vocab)
        model_config.tgt_vocab_size = len(self.tgt_vocab)

        model_config.task_pad_idx = self.task_vocab['<PAD>']
        model_config.src_pad_idx = self.src_vocab['<PAD>']
        model_config.tgt_pad_idx = self.tgt_vocab['<PAD>']
        model_config.n_actions = len(self.tgt_vocab)

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.tgt_pad_idx = model_config.tgt_pad_idx

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=model_config.tgt_pad_idx, reduction='none')

    def receive(self, src_words, tasks, regxes, rewards):
        self.train_src_words = src_words
        self.train_tasks = tasks
        self.train_regexes = regxes
        self.train_rewards = rewards

    def compute_loss(self):

        batch_size = len(self.train_src_words)

        self.init(self.train_src_words, self.train_tasks, False)

        final_rewards = self._to_tensor(self.train_rewards).long()

        normalized_reward_seqs = []

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

            pred_rewards = self.model.predict_baseline().squeeze(1)
            normalized_reward_seqs.append(pred_rewards - final_rewards)

        loss = 0
        zipped_info = zip(self.action_logit_seqs,
                          self.gold_action_seqs,
                          normalized_reward_seqs)

        for logits, golds, rewards in zipped_info:

            masks = golds != self.tgt_pad_idx
            normalizer = masks.sum()

            if normalizer == 0:
                continue

            print(rewards.tolist())

            actor_loss = self.loss_fn(logits, golds) * rewards.detach()
            critic_loss = rewards ** 2
            loss += ((actor_loss + critic_loss) * masks).sum() / normalizer

        print(loss.item())

        return loss

    def learn(self):

        loss = self.compute_loss()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(self.gold_action_seqs)

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = { 'model_state_dict': self.model.state_dict(),
                 'optim_state_dict': self.optim.state_dict() }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        #self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





