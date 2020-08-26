import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import math

import torch

from misc import util
from .imitation import ImitationTrainer


class AbstractLanguageTrainer(ImitationTrainer):

    def do_rollout(self, batch, world, student, teacher, is_eval):

        init_states = []
        tasks = []
        instructions = []

        batch_size = len(batch)
        for item in batch:
            tasks.append(item['task'])
            init_states.append(world.init_state(item['grid'], item['init_pos']))

        student.init(init_states, tasks, is_eval)

        debug_idx = 0
        if debug_idx != -1:
            init_states[debug_idx].render()

        states = init_states[:]
        timer = [self.config.trainer.max_timesteps] * batch_size
        done = [False] * batch_size
        num_interactions = 0
        num_steps = 0
        bits_per_word = math.ceil(math.log(len(self.config.vocab)) / math.log(2))
        #print('bits_per_word', bits_per_word)

        action_seqs = [[] for i in range(batch_size)]
        descriptions = [[] for _ in range(batch_size)]
        description_memory = [{} for _ in range(batch_size)]

        instructions = [None] * batch_size
        starts = [None] * batch_size

        t = 0

        random = self.config.random

        while not all(done):

            if is_eval:

                with torch.no_grad():
                    env_actions = student.act(states, debug_idx=debug_idx)

                """
                if self.config.sanity_check_1 or self.config.sanity_check_2:
                    env_actions = [teacher(tasks[i], states[i]) for i in range(batch_size)]
                """
            else:
                asked_with_instructions = [set() for _ in range(batch_size)]
                env_actions = [student.STOP if done[i] else None
                    for i in range(batch_size)]
                env_action_probs = [1 if done[i] else None
                    for i in range(batch_size)]

                while not all(x is not None for x in env_actions):

                    # Set instruction to execute
                    for i in range(batch_size):
                        if env_actions[i] is not None:
                            instructions[i], starts[i] = ['<PAD>'], None
                        else:
                            # Execute instruction at the top of the stack
                            instructions[i], starts[i] = student.top_stack(i)

                    # Student takes actions
                    with torch.no_grad():
                        actions, ask_actions, action_probs = \
                            student.act(states, debug_idx=debug_idx)

                    for i in range(batch_size):
                        instr = ' '.join(instructions[i])
                        if env_actions[i] is None and instr in asked_with_instructions[i]:
                            ask_actions[i] = 0

                    """
                    if self.config.sanity_check_1:
                        actions = [teacher(tasks[i], states[i]) for i in range(batch_size)]
                        ask_actions = [0] * batch_size

                    if self.config.sanity_check_2:
                        # ask 40% of the time
                        random = self.config.random
                        ask_actions = [int(random.rand() < 0.4) for i in range(batch_size)]
                        # follow instruction optimally
                        for i in range(batch_size):
                            if env_actions[i] is not None:
                                continue
                            actions[i] = teacher(instructions[i], states[i])
                            if instructions[i] in [['left'], ['right'], ['up'], ['down'], ['use'], ['stop']]:
                                try:
                                    stacks[i].pop()
                                    instructions[i], starts[i] = stacks[i][-1]
                                except IndexError:
                                    pass
                    """

                    push_indices = []
                    push_instructions = []

                    for i in range(batch_size):

                        if env_actions[i] is not None:
                            continue

                        if instructions[i] == ['stop']:
                            actions[i] = student.STOP
                            ask_actions[i] = 0
                            student.pop_stack(i)
                            instructions[i], starts[i] = student.top_stack(i)

                        if i == debug_idx:
                            print('ask', ask_actions[debug_idx], 'nav', actions[debug_idx])

                        # Student asks
                        if ask_actions[i] == student.ASK:

                            if i == debug_idx:
                                print(' '.join(instructions[i]))

                            if i == debug_idx and not done[i]:
                                print('=== ASK at', states[i].pos, 'with instruction', instructions[i])

                            asked_with_instructions[i].add(' '.join(instructions[i]))

                            if self.config.trainer.random_describe:
                                ask_description = random.rand() >= action_probs[i]
                            else:
                                ask_description = 1

                            # Ask teacher to describe execution of current instruction
                            if starts[i] < t and ask_description:
                                time_range = (starts[i], t)
                                # Check if description has been cached
                                if time_range in description_memory[i]:
                                    decsr = description_memory[i][time_range]
                                else:
                                    traj = student.slice_trajectory(i, *time_range)
                                    descr = teacher.describe(*traj)
                                    description_memory[i][time_range] = descr

                                    if descr is not None:
                                        for item in descr:
                                            num_interactions += len(item[0]) * bits_per_word

                                # Add to data for training interpreter
                                if descr is not None:
                                    student.add_interpreter_data(descr, traj)
                                    if i == debug_idx and not done[i]:
                                        print('++++ Add descriptions:', end=' ')
                                        for item in descr:
                                            print(item[0], end=' ')
                                        print()

                            # Request teacher a new instruction
                            instr = teacher.instruct(instructions[i], states[i], debug=i==debug_idx)
                            if instr is not None:
                                num_interactions += len(instr) * bits_per_word

                            if i == debug_idx and not done[i]:
                                print('ASKed and receive instruction', instr)

                            if instr is not None:
                                # Defer current instruction, follow new instruction
                                push_indices.append(i)
                                push_instructions.append(instr)

                        elif actions[i] == student.STOP:
                            # Stop executing current instruction
                            student.pop_stack(i)

                            # Stack empty = terminate executing task command
                            if student.is_stack_empty(i):
                                env_actions[i] = actions[i]
                                env_action_probs[i] = action_probs[i]

                            time_range = (starts[i], t)
                            traj = student.slice_trajectory(i, *time_range)

                            instr = teacher.instruct(instructions[i], states[i])

                            if instr is not None:
                                num_interactions += len(instr) * bits_per_word

                            if instr == ['stop']:
                                state_seq, action_seq, action_prob_seq = traj
                                state_seq.append(state_seq[-1])
                                action_seq.append(student.STOP)
                                action_prob_seq.append(1)
                                descr = [(instructions[i], traj)]
                                #student.add_student_data(descr, traj)
                            else:
                                if self.config.trainer.random_describe:
                                    ask_description = random.rand() >= action_probs[i]
                                else:
                                    ask_description = 1

                                if ask_description:
                                    if time_range in description_memory[i]:
                                        descr = description_memory[i][time_range]
                                    else:
                                        descr = teacher.describe(*traj)
                                        description_memory[i][time_range] = descr

                                        if descr is not None:
                                            for item in descr:
                                                num_interactions += len(item[0]) * bits_per_word
                                else:
                                    descr = None

                            if descr is not None:
                                if i == debug_idx and not done[i]:
                                    print('+++ STOPped and add description', [item[0] for item in descr])
                                student.add_interpreter_data(descr, traj)
                        else:
                            env_actions[i] = actions[i]
                            env_action_probs[i] = action_probs[i]

                    student.push_stacks(t, push_indices, push_instructions)

            with torch.no_grad():
                student.decode_stacks(states)

            for i in range(batch_size):

                if not done[i]:
                    _, states[i] = states[i].step(env_actions[i])
                    action_seqs[i].append(env_actions[i])

                    if not is_eval:
                        student.append_trajectory(
                            i, env_actions[i], env_action_probs[i], states[i])

                    num_steps += (not is_eval)

            if debug_idx != -1 and not done[debug_idx]:
                print('ENV:', env_actions[debug_idx])
                states[debug_idx].render()

            for i in range(batch_size):
                timer[i] -= 1
                done[i] |= student.is_stack_empty(i) or \
                           env_actions[i] == student.STOP or timer[i] <= 0

                if done[i]:
                    student.terminate(i)

            t += 1

        if not is_eval:
            student.process_data()

        if debug_idx != -1:
            print(str(tasks[debug_idx]))
            print(batch[debug_idx]['ref_actions'])
            print(action_seqs[debug_idx])
            print()

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

