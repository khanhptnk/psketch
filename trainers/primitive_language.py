import logging
import os
import sys
sys.path.append('..')
import itertools
import json

import torch

from misc import util
from .imitation import ImitationTrainer


class PrimitiveLanguageTrainer(ImitationTrainer):

    def do_rollout(self, batch, world, student, teacher, is_eval):

        init_states = []
        tasks = []
        instructions = []

        batch_size = len(batch)
        for item in batch:
            tasks.append(item['task'])
            init_states.append(world.init_state(item['grid'], item['init_pos']))
            instructions.append(teacher.instruct(world, item['ref_actions']))

        print(init_states[0].render())
        print(batch[0]['ref_actions'], instructions[0])
        student.init(tasks, instructions, init_states, is_eval)

        states = init_states[:]
        #if is_eval:
        timer = [self.config.trainer.max_timesteps] * batch_size
        #else:
        #    timer = [len(item['ref_actions']) for item in batch]
        done = [False] * batch_size
        action_seqs = [[] for i in range(batch_size)]
        state_seqs = [[state] for state in states]
        num_interactions = 0

        t = 0

        while not all(done):

            if is_eval:
                actions = student.act(states, t)
            else:
                actions = student.instructed_act(states, t)

            for i in range(batch_size):

                if not done[i]:
                    _, states[i] = states[i].step(actions[i])
                    action_seqs[i].append(actions[i])
                    state_seqs[i].append(states[i])

                timer[i] -= 1
                done[i] |= actions[i] == world.actions.STOP.index or \
                           timer[i] <= 0
                if done[i]:
                    student.terminate(i)

            t += 1

        if not is_eval:
            descriptions = []
            for i in range(batch_size):
                description = teacher.describe(
                    world, action_seqs[i], state_seqs[i])
                descriptions.append(description)
            print(action_seqs[0], descriptions[0])
            student.receive(descriptions)

            # Decode the second time. This time without exploration
            student.set_instructions(instructions)
            student.reset_history()
            student.instructed_model.eval()

            states = init_states[:]
            timer = [self.config.trainer.max_timesteps] * batch_size
            #timer = [len(item['ref_actions']) for item in batch]
            done = [False] * batch_size
            action_seqs = [[] for i in range(batch_size)]

            t = 0

            while not all(done):

                actions = student.instructed_act(states, t)

                for i in range(batch_size):

                    #actions[i] = teacher(tasks[i], states[i])
                    #student.action_seqs[-1][i] = actions[i]

                    if not done[i]:
                        _, states[i] = states[i].step(actions[i])
                        action_seqs[i].append(actions[i])

                    timer[i] -= 1
                    done[i] |= actions[i] == world.actions.STOP.index or \
                           timer[i] <= 0

                    if done[i]:
                        student.terminate(i)
                t += 1

            for i in range(batch_size):
                #if tuple(action_seqs[i]) != batch[i]['ref_actions']:
                if i == 0:
                    #nit_states[i].render()
                    #print(batch[i]['ref_actions'], instructions[i])
                    print(action_seqs[i], descriptions[i])
                    print()
                    break
            student.imitate_instructed()

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
                'num_interactions': num_interactions
            }

        return info

