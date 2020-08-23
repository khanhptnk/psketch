import logging
import os
import sys
sys.path.append('..')
import itertools
import json

import torch

from misc import util
from .interactive_primitive_language import InteractivePrimitiveLanguageTrainer


class ActivePrimitiveLanguageTrainer(InteractivePrimitiveLanguageTrainer):

    def do_rollout(self, batch, world, student, teacher, is_eval):

        init_states = []
        tasks = []
        instructions = []

        batch_size = len(batch)
        for item in batch:
            tasks.append(item['task'])
            init_states.append(world.init_state(item['grid'], item['init_pos']))

        student.init(init_states)
        student.set_tasks(tasks, is_eval=is_eval)

        #init_states[0].render()
        #print(batch[0]['ref_actions'])

        states = init_states[:]
        timer = [self.config.trainer.max_timesteps] * batch_size
        done = [False] * batch_size
        action_seqs = [[] for i in range(batch_size)]
        ask_action_seqs = [[] for i in range(batch_size)]
        num_interactions = 0
        num_steps = 0

        instructions = [['<PAD>'] for _ in range(batch_size)]
        descriptions = [['<PAD>'] for _ in range(batch_size)]

        while not all(done):

            if is_eval:
                actions = student.act(states)
            else:
                actions, ask_actions = student.act(states)

                for i in range(batch_size):
                    if ask_actions[i] == student.ASK:
                        instructions[i] = teacher(tasks[i], states[i])
                        num_interactions += (not is_eval and not done[i])

                student.set_instructions(instructions, is_eval=False)
                instructed_actions = student.instructed_act(states, ask_actions)

                for i in range(batch_size):
                    if ask_actions[i] == student.ASK:
                        actions[i] = instructed_actions[i]

            prev_states = states[:]

            for i in range(batch_size):

                if not done[i]:
                    _, states[i] = states[i].step(actions[i])
                    action_seqs[i].append(actions[i])
                    num_steps += (not is_eval)

                    # Teacher describes student actions
                    if not is_eval:
                        ask_action_seqs[i].append(ask_actions[i])
                        if ask_actions[i] == student.ASK:
                            descriptions[i] = teacher.describe(world,
                                [actions[i]], [prev_states[i], states[i]])

            if not is_eval:
                student.receive(descriptions, ask_actions)

            for i in range(batch_size):
                timer[i] -= 1
                done[i] |= actions[i] == world.actions.STOP.index or \
                           timer[i] <= 0
                if done[i]:
                    student.terminate(i)

        if not is_eval:
            student.set_tasks(tasks, is_eval)
            student.imitate_instructed()

        #print(action_seqs[0])
        #print(ask_action_seqs[0])
        #print()

        success = []
        distances = []
        for i in range(batch_size):
            success.append(states[i].satisfies(tasks[i]))
            if tasks[i].goal_name == 'get':
                if not success[i]:
                    state = world.init_state(
                        batch[i]['grid'], states[i].pos, states[i].dir)
                    _, best_action_seq = teacher.find_closest_resources(
                        tasks[i], state)
                    distances.append(len(best_action_seq))
                else:
                    distances.append(0)

        info = {
                'action_seqs': action_seqs,
                'success': success,
                'distances': distances,
                'num_interactions': num_interactions,
                'num_steps': num_steps
            }

        return info

