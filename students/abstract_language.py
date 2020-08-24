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
        self.POP = -1

    def terminate(self, i):
        self.terminated[i] = True

    def _encode_and_pad(self, xs):
        # Init instructed model
        encodings = []
        masks = []
        for x in xs:
            encodings.append([self.vocab[w] for w in x])
            masks.append([0] * len(encodings[-1]))
        # Padding
        #max_len = max([len(encoding) for encoding in encodings])
        max_len = 3

        for encoding in encodings:
            encoding.extend([self.vocab['<PAD>']] * (max_len - len(encoding)))

        for mask in masks:
            mask.extend([1] * (max_len - len(mask)))

        encodings = self._to_tensor(encodings).long()
        masks = self._to_tensor(masks).bool()

        return encodings, masks

    def init_model(self, model, instructions):
        instruction_encodings, instruction_masks = \
            self._encode_and_pad(instructions)
        return model.encode(instruction_encodings, src_mask=instruction_masks)

    def init(self, states, tasks, is_eval):

        if is_eval:
            self.interpreter_model.eval()
            self.student_model.eval()
            self.acting_model = self.student_model
            if self.config.trainer.test_interpreter:
                self.acting_model = self.interpreter_model
        else:
            self.student_model.train()
            self.interpreter_model.train()
            self.acting_model = self.interpreter_model

        self.batch_size = len(states)

        self.state_seqs = [[state] for state in states]
        self.action_seqs = [[] for _ in range(self.batch_size)]
        self.action_prob_seqs = [[] for _ in range(self.batch_size)]

        self.terminated = [False] * self.batch_size

        self.interpreter_data = []
        self.student_data = []

        self.instruction_stacks = [[] for _ in range(self.batch_size)]
        self.state_stacks = [[] for _ in range(self.batch_size)]

        self.push_stacks(0, list(range(self.batch_size)),
            [['<PAD>'] for _ in range(self.batch_size)])

        self.push_stacks(0, list(range(self.batch_size)),
            [str(task).split() for task in tasks])

    def push_stacks(self, time, indices, instructions):

        assert len(indices) == len(instructions)

        if not indices:
            return

        c, m, (h0, h1), t = self.init_model(self.acting_model, instructions)

        c_list = c.split(1, dim=0)
        m_list = m.split(1, dim=0)
        h0_list = h0.split(1, dim=1)
        h1_list = h1.split(1, dim=1)
        t_list = t.split(1, dim=0)
        for i, (idx, instr) in enumerate(zip(indices, instructions)):
            self.instruction_stacks[idx].append((instr, time))
            self.state_stacks[idx].append(
                (c_list[i], m_list[i], h0_list[i], h1_list[i], t_list[i]))

    def top_stack(self, i):
        return self.instruction_stacks[i][-1]

    def pop_stack(self, i):
        self.instruction_stacks[i].pop()
        self.state_stacks[i].pop()

    def is_stack_empty(self, i):
        return len(self.instruction_stacks[i]) <= 1

    def get_model_states(self, items):

        c_list = []
        m_list = []
        h0_list = []
        h1_list = []
        t_list = []

        for item in items:
            c, m, h0, h1, t = item
            c_list.append(c)
            m_list.append(m)
            h0_list.append(h0)
            h1_list.append(h1)
            t_list.append(t)

        c = torch.cat(c_list, dim=0)
        m = torch.cat(m_list, dim=0)
        h0 = torch.cat(h0_list, dim=1)
        h1 = torch.cat(h1_list, dim=1)
        t = torch.cat(t_list, dim=0)

        return c, m, (h0, h1), t

    def act(self, states, debug_idx=None):

        s = [state.features() for state in states]
        s = self._to_tensor(s).float()

        stack_items = []
        for stack in self.state_stacks:
            stack_items.append(stack[-1])

        instructions = []
        for stack in self.instruction_stacks:
            instructions.append(stack[-1][0])

        c, m, h, t = self.get_model_states(stack_items)
        action_logits, _, _ = self.acting_model.decode(s, h, t, c, m)
        action_dists = D.Categorical(logits=action_logits)

        if self.acting_model.training:
            actions = action_dists.sample()
            entropies = action_dists.entropy() / math.log(self.n_actions)
            ask_actions = (entropies > self.uncertainty_threshold).long()

            if debug_idx != -1 and instructions[debug_idx] != ['<PAD>']:
                print(instructions[debug_idx], action_dists.probs[debug_idx], entropies.tolist()[debug_idx])

            for i in range(self.batch_size):
                if self.terminated[i]:
                    actions[i] = self.STOP
                    ask_actions[i] = 0

            action_probs = action_dists.probs[range(len(actions)), actions]

            return actions.tolist(), ask_actions.tolist(), action_probs.tolist()

        if debug_idx != -1 and instructions[debug_idx] != ['<PAD>']:
            print(instructions[debug_idx], action_dists.probs[debug_idx])

        actions = action_logits.max(dim=1)[1]

        return actions.tolist()

    def decode_stacks(self, states):

        s = []
        stack_items = []
        for i, stack in enumerate(self.state_stacks):
            stack_items.extend(stack)
            s.extend([states[i].features() for _ in range(len(stack))])

        s = self._to_tensor(s).float()
        c, m, h, t = self.get_model_states(stack_items)
        _, h, t = self.acting_model.decode(s, h, t, c, m)

        h0, h1 = h

        h0_list_iter = iter(h0.split(1, dim=1))
        h1_list_iter = iter(h1.split(1, dim=1))
        t_list_iter = iter(t.split(1, dim=0))

        for stack in self.state_stacks:
            for i in range(len(stack)):
                item = stack[i]
                stack[i] = (item[0], item[1],
                            next(h0_list_iter),
                            next(h1_list_iter),
                            next(t_list_iter))

    def add_interpreter_data(self, description, trajectory):
        #self.interpreter_data.append((description, trajectory))
        self.interpreter_data.extend(description)

    def add_student_data(self, description, trajectory):
        #self.student_data.append((description, trajectory))
        self.student_data.extend(description)

    def append_trajectory(self, i, action, action_prob, state):
        self.action_seqs[i].append(action)
        self.action_prob_seqs[i].append(action_prob)
        self.state_seqs[i].append(state)

    def slice_trajectory(self, i, start, end, return_prob=False):

        # (s_start, a_start, ..., a_{end - 1}, state)
        state_seq = self.state_seqs[i][start:(end + 1)]
        action_seq = self.action_seqs[i][start:end]
        action_prob_seq = self.action_prob_seqs[i][start:end]

        return state_seq, action_seq, action_prob_seq

    def _make_batch(self, data):

        def wrap_state_seq(data):
            max_len = max([len(item) - 1 for item in data])
            data = [item[:-1] + [item[-1]] * (max_len - len(item) + 1)
                for item in data]
            return data

        def wrap_action_seq(data):
            max_len = max([len(item) for item in data])
            data = [item + [-1] * (max_len - len(item)) for item in data]
            data = self._to_tensor(data).transpose(0, 1).long()
            return data

        def wrap_prob_seq(data):
            max_len = max([len(item) for item in data])
            data = [item + [-1] * (max_len - len(item)) for item in data]
            data = self._to_tensor(data).transpose(0, 1).float()
            return data

        self.random.shuffle(data)
        batched_data = []
        i = 0
        while i < len(data):
            j = i + self.batch_size
            batch = data[i:j]
            d_batch, t_batch = zip(*batch)
            s_batch, a_batch, p_batch = zip(*t_batch)
            s_batch = wrap_state_seq(s_batch)
            a_batch = wrap_action_seq(a_batch)
            p_batch = wrap_prob_seq(p_batch)
            batched_data.append((d_batch, s_batch, a_batch, p_batch))
            i = j
        return batched_data

    def _compute_loss(self, data, model, weights,
            weight_target_by_uncertainty=False):

        for instructions, state_seqs, action_seqs, action_prob_seqs in data:

            loss_weights = []
            for instr in instructions:
                instr = ' '.join(instr)
                loss_weights.append(weights[instr])
            loss_weights = self._to_tensor(loss_weights).float()

            c, m, h, t = self.init_model(model, instructions)

            loss = 0
            seq_len = action_seqs.shape[0]

            zipped_info = enumerate(zip(action_seqs, action_prob_seqs))
            for i, (targets, target_probs) in zipped_info:

                states = [state_seq[i] for state_seq in state_seqs]
                s = [state.features() for state in states]
                s = self._to_tensor(s).float()
                logits, h, t = model.decode(s, h, t, c, m)

                losses = self.loss_fn(logits, targets)
                valid_targets = (targets != -1)
                losses = losses * loss_weights * valid_targets

                if weight_target_by_uncertainty:
                    losses = losses * target_probs

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

        self.interpreter_data = self._filter_data(self.interpreter_data)
        self.student_data = self._filter_data(self.student_data)

        self.interpreter_weights = self._compute_weights(self.interpreter_data)
        self.student_weights = self._compute_weights(self.student_data)

        self.interpreter_data = self._make_batch(self.interpreter_data)
        self.student_data = self._make_batch(self.student_data)

    def learn(self):

        i_loss_generator = self._compute_loss(self.interpreter_data,
            self.interpreter_model, self.interpreter_weights,
            weight_target_by_uncertainty=self.config.student.weight_interpreter_target_by_uncertainty)
        s_loss_generator = self._compute_loss(self.student_data,
            self.student_model, self.student_weights,
            weight_target_by_uncertainty=self.config.student.weight_student_target_by_uncertainty)

        total_i_loss = []
        for i_loss, i_len in i_loss_generator:
            self.optim.zero_grad()
            i_loss.backward()
            self.optim.step()
            total_i_loss.append(i_loss.item() / i_len)

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

        if total_s_loss:
            total_s_loss = sum(total_s_loss) / len(total_s_loss)
        else:
            total_s_loss = 0

        return total_i_loss + total_s_loss

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





