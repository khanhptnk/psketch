import os
import sys
import json
import logging
import math
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.distributions as D

import models
from .primitive_language import PrimitiveLanguageStudent


class AbstractLanguageStudent(PrimitiveLanguageStudent):

    ASK = 1

    def __init__(self, config):

        self.config = config
        self.random = config.random
        self.vocab = config.vocab
        self.device = config.device
        self.uncertainty_threshold = config.student.uncertainty_threshold

        model_config = config.student.model
        model_config.device = config.device
        model_config.vocab_size = len(config.vocab)
        model_config.pad_idx = config.vocab['<PAD>']
        model_config.enc_hidden_size = config.student.model.hidden_size
        model_config.dec_hidden_size = config.student.model.hidden_size

        self.interpreter_model = models.load(model_config).to(self.device)
        self.student_model = models.load(model_config).to(self.device)

        logging.info('interpreter_model: ' + str(self.interpreter_model))
        logging.info('student_model: ' + str(self.student_model))

        lr = model_config.learning_rate
        self.optim = torch.optim.AdamW(
            list(self.interpreter_model.parameters()) + \
            list(self.student_model.parameters()), lr=lr)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    def prepare(self, world):
        self.world = world
        self.STOP = world.actions.STOP.index
        self.n_actions = world.n_actions

    """
    def set_instructions(self, model, instructions):

        instruction_encodings, instruction_mask = \
            self._encode_and_pad(instructions)
        model.init(instruction_encodings, src_mask=instruction_mask)

    def advance_time(self):
        if self.interpreter_model.training:
            self.interpreter_model.advance_time()
        else:
            self.student_model.advance_time()
    """

    def terminate(self, i):
        self.terminated[i] = True

    def init_model(self, model, instructions):
        instruction_encodings, instruction_masks = \
            self._encode_and_pad(instructions)
        init_h = model.encode(instruction_encodings, src_mask=instruction_masks)
        init_t = self._to_tensor([0] * len(instructions)).long()
        return init_h, init_t

    def init(self, states, tasks, is_eval):

        self.batch_size = len(states)

        self.state_seqs = [[state] for state in states]
        self.action_seqs = [[] for _ in range(self.batch_size)]

        self.terminated = [False] * self.batch_size

        self.interpreter_data = []
        self.student_data = []

        if is_eval:
            self.interpreter_model.eval()
            self.student_model.eval()
            tasks = [str(task).split() for task in tasks]
            self.student_h, self.student_t = self.init_model(self.student_model, tasks)
        else:
            self.student_model.train()
            self.interpreter_model.train()

    def act(self, states):
        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        action_logits, self.student_h, self.student_t = \
            self.student_model.decode(
                state_features, self.student_h, self.student_t)

        if self.student_model.training:
            actions = D.Categorical(logits=action_logits).sample()
        else:
            actions = action_logits.max(dim=1)[1]

        return actions.tolist()

    def interpret(self, states, instructions, debug_idx=0):

        h, t = self.init_model(self.interpreter_model, instructions)

        if all(self.interpreter_reset_at_index):
            self.interpreter_h, self.interpreter_t = h, t
        else:
            for i in range(self.batch_size):
                if self.interpreter_reset_at_index[i]:
                    self.interpreter_t[i] = t[i]
                    for j in range(2):
                        self.interpreter_h[j][:, i, :] = h[j][:, i, :]

        state_features = [state.features() for state in states]
        state_features = self._to_tensor(state_features).float()

        action_logits, self.next_interpreter_h, self.next_interpreter_t = \
            self.interpreter_model.decode(
                state_features, self.interpreter_h, self.interpreter_t)

        if self.interpreter_model.training:
            action_dists = D.Categorical(logits=action_logits)
            actions = action_dists.sample()
            entropies = action_dists.entropy() / math.log(self.n_actions)
            ask_actions = (entropies > self.uncertainty_threshold).long()

            if debug_idx != -1 and instructions[debug_idx] != ['<PAD>']:
                print(instructions[debug_idx], action_dists.probs[debug_idx], entropies.tolist()[debug_idx], actions.tolist()[debug_idx], self.interpreter_t.tolist()[debug_idx])

            #print(actions[0])

            for i in range(self.batch_size):
                if self.terminated[i]:
                    actions[i] = self.STOP
                    ask_actions[i] = 0

            return actions.tolist(), ask_actions.tolist()

        actions = action_logits.max(dim=1)[1]
        return actions.tolist()

    def advance_interpreter_state(self):
        self.interpreter_h = self.next_interpreter_h
        self.interpreter_t = self.next_interpreter_t

    def add_interpreter_data(self, description, trajectory):
        #self.interpreter_data.append((description, trajectory))
        self.interpreter_data.extend(description)

    def add_student_data(self, description, trajectory):
        #self.student_data.append((description, trajectory))
        self.student_data.extend(description)

    def append_trajectory(self, i, action, state):
        self.action_seqs[i].append(action)
        self.state_seqs[i].append(state)

    def slice_trajectory(self, i, start, end):

        # (s_start, a_start, ..., a_{end - 1}, state)
        state_seq = self.state_seqs[i][start:(end + 1)]
        action_seq = self.action_seqs[i][start:end]

        return state_seq, action_seq

    def _make_batch(self, data):

        def wrap_state_seq(data):
            """
            for item in data:
                state_feature_seq = []
                for state in item[:-1]:
                    state_feature_seq.append(state.features())
                state_feature_seqs.append(state_feature_seq)

            max_len = max([len(item) for item in state_feature_seqs])

            state_feature_seqs = [item + [item[-1]] * (max_len - len(item))
                for item in state_feature_seqs]
            state_feature_seqs = self._to_tensor(state_feature_seqs)\
                .transpose(0, 1).contiguous().float()
            """


            max_len = max([len(item) - 1 for item in data])
            data = [item[:-1] + [item[-1]] * (max_len - len(item) + 1)
                for item in data]
            return data

        def wrap_action_seq(data):
            max_len = max([len(item) for item in data])
            data = [item + [-1] * (max_len - len(item)) for item in data]
            data = self._to_tensor(data).transpose(0, 1).long()
            return data

        #data = sorted(data, key=lambda x: len(x[1][1]))
        self.random.shuffle(data)
        batched_data = []
        i = 0
        while i < len(data):
            j = i + self.batch_size
            batch = data[i:j]
            d_batch, t_batch = zip(*batch)
            s_batch, a_batch = zip(*t_batch)
            s_batch = wrap_state_seq(s_batch)
            a_batch = wrap_action_seq(a_batch)
            #assert s_batch.shape[0] == a_batch.shape[0]
            batched_data.append((d_batch, s_batch, a_batch))
            i = j
        return batched_data

    def _compute_loss(self, data, model, weights):

        for instructions, state_seqs, action_seqs in data:

            loss_weights = []
            for instr in instructions:
                instr = ' '.join(instr)
                loss_weights.append(weights[instr])
            loss_weights = self._to_tensor(loss_weights).float()
            #print('loss weights', loss_weights.shape)

            h, t = self.init_model(model, instructions)

            loss = 0
            seq_len = action_seqs.shape[0]

            #print(instructions[0], action_seqs[:, 0].tolist())

            for i, targets in enumerate(action_seqs):

                states = [state_seq[i] for state_seq in state_seqs]
                state_features = [state.features() for state in states]
                state_features = self._to_tensor(state_features).float()
                logits, h, t = model.decode(state_features, h, t)

                losses = self.loss_fn(logits, targets)
                valid_targets = (targets != -1)
                losses = losses * loss_weights * valid_targets
                loss += losses.sum() / valid_targets.sum()

            yield loss, seq_len

    def _compute_weights(self, data):

        weights = {}
        instructions = []
        for item in data:
            instr = ' '.join(item[0])
            instructions.append(instr)
        counter = Counter(instructions)
        for instr in counter:
            if self.config.student.reweight_data:
                weights[instr] = 1 / counter[instr]
            else:
                weights[instr] = 1
        return weights

    def _filter_data(self, data):
        data_split = defaultdict(list)
        for item in data:
            instr = ' '.join(item[0])
            data_split[instr].append(item)
        limit = self.batch_size
        new_data = []
        for instr in data_split:
            new_data.extend(data_split[instr][:limit])
        return new_data

    def process_data(self):

        #for item in self.student_data:
            #print('--------', item[0])
        #if self.student_data:
            #item = self.student_data[-1]
            #item[1][0][0].render()
            #item[1][0][-1].render()
            #print(item[0], item[1][1])

        self.interpreter_data = self._filter_data(self.interpreter_data)
        self.student_data = self._filter_data(self.student_data)

        self.interpreter_weights = self._compute_weights(self.interpreter_data)
        self.student_weights = self._compute_weights(self.student_data)

        #print(len(self.interpreter_data), len(self.student_data))
        self.interpreter_data = self._make_batch(self.interpreter_data)
        self.student_data = self._make_batch(self.student_data)

    def learn(self):

        i_loss_generator = self._compute_loss(self.interpreter_data,
            self.interpreter_model, self.interpreter_weights)
        s_loss_generator = self._compute_loss(self.student_data,
            self.student_model, self.student_weights)

        total_i_loss = []
        for i_loss, i_len in i_loss_generator:
            self.optim.zero_grad()
            i_loss.backward()
            self.optim.step()
            total_i_loss.append(i_loss.item() / i_len)
            #print(total_i_loss[-1])

        #print(len(total_i_loss))
        if total_i_loss:
            total_i_loss = sum(total_i_loss) / len(total_i_loss)
        else:
            total_i_loss = 0

        total_s_loss = []
        for s_loss, s_len in s_loss_generator:
            self.optim.zero_grad()
            s_loss.backward()
            self.optim.step()
            total_s_loss.append(s_loss.item() / s_len)

        #print(len(total_s_loss))
        if total_s_loss:
            total_s_loss = sum(total_s_loss) / len(total_s_loss)
        else:
            total_s_loss = 0

        return 0

        #return total_i_loss + total_s_loss

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = {
            'interpreter_model_state_dict': self.interpreter_model.state_dict(),
            'student_model_state_dict': self.student_model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        self.interpreter_model.load_state_dict(ckpt['interpreter_model_state_dict'])
        self.student_model.load_state_dict(ckpt['student_model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





