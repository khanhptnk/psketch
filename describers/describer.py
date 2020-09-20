import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import editdistance

import models

from models.beam import Beam

class Describer(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.src_vocab = config.char_vocab
        self.tgt_vocab = config.word_vocab

        model_config = config.describer.model
        model_config.device = config.device
        model_config.src_vocab_size = len(self.src_vocab)
        model_config.tgt_vocab_size = len(self.tgt_vocab)

        model_config.src_pad_idx = self.src_vocab['<PAD>']
        model_config.tgt_pad_idx = self.tgt_vocab['<PAD>']
        model_config.n_actions = len(self.tgt_vocab)

        model_config.enc_hidden_size = model_config.hidden_size
        model_config.dec_hidden_size = model_config.hidden_size

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.tgt_pad_idx)
        self.n_samples = self.config.describer.n_samples

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _pad(self, seq, pad_token, max_len):
        return seq.extend([pad_token] * (max_len - len(seq)))

    def _index_and_pad(self, examples, vocab):

        encodings = []
        max_len = 0
        for x in examples:
            encodings.append([])
            for seq in x:
                indices = [vocab[w] for w in seq]
                encodings[-1].append(indices)
                max_len = max(max_len, len(indices))

        for x in encodings:
            for seq in x:
                self._pad(seq, vocab['<PAD>'], max_len)

        encodings = self._to_tensor(encodings).long()

        return encodings

    def init(self, examples, is_eval):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval
        self.batch_size = len(examples)

        self.pred_action_seqs = [['<'] for _ in range(self.batch_size)]
        self.gold_action_seqs = []
        self.action_logit_seqs = []
        self.terminated = [False] * self.batch_size

        example_encodings = self._index_and_pad(examples, self.src_vocab)
        self.model.encode(example_encodings)

        self.prev_actions = [self.tgt_vocab['<']] * self.batch_size
        self.prev_actions = self._to_tensor(self.prev_actions)

        self.timer = self.config.describer.max_timesteps

    def act(self, gold_actions=None, sample=False):

        action_logits = self.model.decode(self.prev_actions)
        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            if sample:
                pred_actions = D.Categorical(logits=action_logits).sample()
            else:
                pred_actions = action_logits.max(dim=1)[1]
            self.prev_actions = pred_actions

            pred_actions = pred_actions.tolist()
            for i in range(self.batch_size):
                if not self.terminated[i]:
                    w = self.tgt_vocab.get(pred_actions[i])
                    self.pred_action_seqs[i].append(w)
        else:
            gold_actions = [self.tgt_vocab[w] for w in gold_actions]
            gold_actions = self._to_tensor(gold_actions).long()
            self.gold_action_seqs.append(gold_actions)
            self.prev_actions = gold_actions

        self.timer -= 1

        for i in range(self.batch_size):
            self.terminated[i] |= self.timer <= 0
            self.terminated[i] |= self.prev_actions[i].item() == self.tgt_vocab['>']

    def has_terminated(self):
        return all(self.terminated)

    def get_action_seqs(self):
        return self.pred_action_seqs

    """
    def pragmatic_predict(self, src_words, tgt_words, executor, n_samples):

        batch_size = len(src_words)

        words = []
        for src_word, tgt_word in zip(src_words, tgt_words):
            for _ in range(n_samples):
                words.append(src_word + ['@'] + tgt_word)

        self.init(words, True)

        beams = [Beam(n_samples, self.tgt_vocab, self.device, n_best=n_samples)
                    for _ in range(batch_size)]

        while self.timer > 0:
            prev_actions = torch.cat([b.get_last_token() for b in beams], dim=0)
            action_logits = self.model.decode(prev_actions)
            action_ldists = action_logits.log_softmax(dim=1)
            action_ldists = action_logits.view(batch_size, n_samples, -1)

            prev_pos = []
            for i, (b, scores) in enumerate(zip(beams, action_ldists)):
                b.advance(scores)
                offset = n_samples * i
                prev_pos.append(offset + b.get_last_pointer())

            prev_pos = torch.cat(prev_pos, dim=0)
            self.model.index_select_decoder_state(prev_pos)

            self.timer -= 1
            if self.timer <= 0 or all([b.done() for b in beams]):
                break

        all_steps, all_last_positions = [], []
        for b in beams:
            _, steps, last_positions = b.sort_finished()
            all_steps.append(steps)
            all_last_positions.append(last_positions)

        best_pred_tasks = [None] * batch_size
        best_pred_tgt_words = [None] * batch_size
        for i in range(n_samples):
            pred_tasks = []
            zipped_info = zip(beams, all_steps, all_last_positions)
            for b, steps, last_positions in zipped_info:
                seq = b.get_seq(steps[i], last_positions[i])
                pred_tasks.append(seq)
            pred_tgt_words = executor.predict(src_words, pred_tasks)

            for j in range(batch_size):
                if best_pred_tasks[j] is None or pred_tgt_words[j] == tgt_words[j]:
                    best_pred_tasks[j] = pred_tasks[j]
                    best_pred_tgt_words[j] = pred_tgt_words[j]

        scores = [pred == gold
            for pred, gold in zip(best_pred_tgt_words, tgt_words)]

        print(best_pred_tasks[0])
        print('src ', ''.join(src_words[0]))
        print('tgt ', ''.join(tgt_words[0]))
        print('pred', ''.join(best_pred_tgt_words[0]))
        print()

        return best_pred_tasks, scores

    """

    def predict(self, examples):

        self.init(examples, True)

        while not self.has_terminated():
            self.act()

        return self.get_action_seqs()


    def pragmatic_predict(self, examples, executor, n_samples=None):

        def infer_tgt_words(src_words, tgt_words, tasks):
            scores = [0] * batch_size
            pred_tgt_words = [[None] * n_examples for _ in range(batch_size)]
            for j in range(n_examples):
                batch_src_words = [item[j] for item in src_words]
                batch_tgt_words = executor.predict(batch_src_words, tasks)
                for i in range(batch_size):
                    gold = tgt_words[i][j]
                    pred = ''.join(batch_tgt_words[i])
                    scores[i] += gold == pred
                    pred_tgt_words[i][j] = batch_tgt_words[i]
            return scores, pred_tgt_words

        if n_samples is None:
            n_samples = self.n_samples

        batch_size = len(examples)
        n_examples = len(examples[0])

        src_words = []
        tgt_words = []
        for item in examples:
            src_words.append([])
            tgt_words.append([])
            for word in item:
                src_word, tgt_word = word.split('@')
                src_words[-1].append(src_word)
                tgt_words[-1].append(tgt_word)

        best_pred_tasks = self.predict(examples)
        best_scores, best_pred_tgt_words = infer_tgt_words(
            src_words, tgt_words, best_pred_tasks)

        for k in range(n_samples):

            self.init(examples, True)
            while not self.has_terminated():
                self.act(sample=True)
            pred_tasks = self.get_action_seqs()

            curr_scores, curr_pred_tgt_words = infer_tgt_words(
                src_words, tgt_words, pred_tasks)

            for i in range(batch_size):
                if curr_scores[i] > best_scores[i]:
                    best_scores[i] = curr_scores[i]
                    best_pred_tasks[i] = pred_tasks[i]
                    best_pred_tgt_words[i] = curr_pred_tgt_words[i]

        return best_pred_tasks, best_pred_tgt_words, best_scores

    def compute_loss(self):
        loss = 0
        zipped_info = zip(self.action_logit_seqs, self.gold_action_seqs)
        for logits, golds in zipped_info:
            loss += self.loss_fn(logits, golds)
        return loss

    def learn(self):

        assert len(self.gold_action_seqs) == len(self.action_logit_seqs)

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
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





