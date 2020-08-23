import logging
import os
import sys
sys.path.append('..')
import itertools
import json

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
        student.interpreter_reset_at_index = [True] * batch_size

        debug_idx = 8
        #init_states[debug_idx].render()

        states = init_states[:]
        timer = [self.config.trainer.max_timesteps] * batch_size
        done = [False] * batch_size
        num_interactions = 0
        num_steps = 0

        action_seqs = [[] for i in range(batch_size)]
        descriptions = [[] for _ in range(batch_size)]
        description_memory = [{} for _ in range(batch_size)]

        instructions = [None] * batch_size
        starts = [None] * batch_size

        stacks = [[(str(task).split(), 0)] for task in tasks]
        t = 0


        while not all(done):

            #states[0].render()
            #print(stacks[0])

            if is_eval:

                if self.config.trainer.test_interpreter:
                    instructions = [str(task).split() for task in tasks]
                    env_actions = student.interpret(states, instructions)
                    student.interpreter_reset_at_index = [False] * batch_size
                else:
                    env_actions = student.act(states)

                """
                if self.config.sanity_check_1 or self.config.sanity_check_2:
                    env_actions = [teacher(tasks[i], states[i]) for i in range(batch_size)]
                """
            else:
                asked_with_instructions = [set() for _ in range(batch_size)]
                env_actions = [student.STOP if done[i] else None
                    for i in range(batch_size)]

                while not all(x is not None for x in env_actions):

                    # Set instruction to execute
                    for i in range(batch_size):
                        if env_actions[i] is not None:
                            instructions[i], starts[i] = ['<PAD>'], None
                        else:
                            # Execute instruction at the top of the stack
                            instructions[i], starts[i] = stacks[i][-1]

                    #student.set_instructions(instructions, is_eval)

                    # Student takes actions
                    actions, ask_actions = student.interpret(
                        states, instructions, debug_idx=debug_idx)

                    #print(asked_with_instructions[debug_idx])
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

                    #print(stacks[0], env_actions[0], actions[0], ask_actions[0])

                    #print('ask', ask_actions[debug_idx], actions[debug_idx], env_actions[debug_idx])

                    student.interpreter_reset_at_index = [False] * batch_size

                    for i in range(batch_size):

                        if env_actions[i] is not None:
                            continue

                        if instructions[i] == ['stop']:
                            actions[i] = student.STOP
                            ask_actions[i] = 0
                            stacks[i].pop()
                            instructions[i], starts[i] = stacks[i][-1]

                        # Student asks
                        if ask_actions[i] == student.ASK:

                            #if i == debug_idx:
                                #print(' '.join(instructions[i]))

                            #if i == debug_idx and not done[i]:
                                #print('=== ASK at', states[i].pos, 'with instruction', instructions[i])


                            asked_with_instructions[i].add(' '.join(instructions[i]))

                            # Ask teacher to describe execution of current instruction
                            if starts[i] < t:
                                time_range = (starts[i], t)
                                # Check if description has been cached
                                if time_range in description_memory[i]:
                                    decsr = description_memory[i][time_range]
                                else:
                                    traj = student.slice_trajectory(i, *time_range)
                                    descr = teacher.describe(*traj)
                                    description_memory[i][time_range] = descr

                                # Add to data for training interpreter
                                if descr is not None:
                                    if i == debug_idx and not done[i]:
                                        state_seq, action_seq = traj
                                        """
                                        state_seq[0].render()
                                        state_seq[-1].render()
                                        """
                                        #print('+++ ASKed and add description', [item[0] for item in descr])
                                        #print('+++ ASKed and add description')
                                        #for item in descr:
                                            #print(item[0], item[1][1])
                                            #item[1][0][0].render()
                                            #item[1][0][-1].render()
                                            #print('------------------------')
                                        #print('Action seq', action_seq)
                                    student.add_interpreter_data(descr, traj)

                            # Request teacher a new instruction
                            instr = teacher.instruct(instructions[i], states[i], debug=i==debug_idx)

                            #if i == debug_idx and not done[i]:
                                #print('ASKed and receive instruction', instr)

                            if instr is None:
                                """
                                student.interpreter_reset_at_index[i] = True
                                # Teacher can't help
                                # Try asking with a higher-level instruction
                                stacks[i].pop()
                                if not stacks[i]:
                                    env_actions[i] = student.STOP
                                """
                                pass
                            else:
                                student.interpreter_reset_at_index[i] = True
                                # Defer current instruction, follow new instruction
                                stacks[i].append((instr, t))
                        elif actions[i] == student.STOP:
                            student.interpreter_reset_at_index[i] = True
                            # Stop executing current instruction
                            stacks[i].pop()

                            # Stack empty = terminate executing task command
                            if not stacks[i]:
                                env_actions[i] = actions[i]

                            time_range = (starts[i], t)
                            traj = student.slice_trajectory(i, *time_range)
                            if teacher.should_stop(instructions[i], states[i]):
                                state_seq, action_seq = traj
                                state_seq.append(state_seq[-1])
                                action_seq.append(student.STOP)
                                #if i == debug_idx and not done[i]:
                                    #print('===>', instructions[i])

                                #print(instructions[i])

                                descr = [(instructions[i], traj)]
                                student.add_student_data(descr, traj)
                            else:
                                if time_range in description_memory[i]:
                                    descr = description_memory[i][time_range]
                                else:
                                    descr = teacher.describe(*traj)
                                    description_memory[i][time_range] = descr

                            if descr is not None:
                                #if i == debug_idx and not done[i]:
                                    #print('+++ STOPped and add description', [item[0] for item in descr])
                                student.add_interpreter_data(descr, traj)
                        else:
                            env_actions[i] = actions[i]

            student.advance_interpreter_state()


            for i in range(batch_size):

                if not done[i]:
                    _, states[i] = states[i].step(env_actions[i])
                    action_seqs[i].append(env_actions[i])
                    student.append_trajectory(i, env_actions[i], states[i])
                    num_steps += (not is_eval)

            #print('ENV:', env_actions[debug_idx])
            #states[debug_idx].render()

            for i in range(batch_size):
                timer[i] -= 1
                done[i] |= not stacks[i] or env_actions[i] == student.STOP or timer[i] <= 0

                if done[i]:
                    student.terminate(i)

            t += 1

        if not is_eval:
            student.process_data()

        #print(str(tasks[debug_idx]))
        #print(batch[debug_idx]['ref_actions'])
        #print(action_seqs[debug_idx])
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

