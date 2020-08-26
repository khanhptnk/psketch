import os
import sys
from collections import defaultdict

from .base import BaseTeacher
from .demonstration import DemonstrationTeacher


class AbstractLanguageTeacher(DemonstrationTeacher):

    PRIMITIVE_ACTION_LEN = 4

    def __init__(self, config):

        super(AbstractLanguageTeacher, self).__init__(config)
        self.student_action_map = {}
        self.random = config.random

        self.task_map = defaultdict(lambda: defaultdict(lambda: None))
        for task in config.task_manager.tasks:
            self.task_map[task.goal_arg][task.goal_name] = task

        self.student_action_map = {}
        self.actions = ['left', 'right', 'up', 'down', 'use', 'stop']

    def action_to_word(self, action, world):
        if action == world.actions.UP.index:
            return 'up'
        elif action == world.actions.DOWN.index:
            return 'down'
        elif action == world.actions.LEFT.index:
            return 'left'
        elif action == world.actions.RIGHT.index:
            return 'right'
        elif action == world.actions.USE.index:
            return 'use'
        assert action == world.actions.STOP.index
        return 'stop'

    def find_next_subtask(self, task, state, debug=False):

        if state.satisfies(task):
            return None

        if task.subtasks is None:
            return task

        for subtask in task.subtasks[:-1]:
            next_subtask = self.find_next_subtask(subtask, state)
            if next_subtask is not None:
                return subtask

        last_subtask = task.subtasks[-1]
        next_subtask = self.find_next_subtask(last_subtask, state)
        assert next_subtask is not None
        return last_subtask

    def instruct(self, instruction, state, debug=False):

        all_primitives = True
        for w in instruction:
            if w not in self.actions:
                all_primitives = False
                break

        if all_primitives:
            return None

        if self.config.teacher.primitive_language_only:
            goal_name, goal_arg = instruction
            task = self.task_map[goal_arg][goal_name]
            assert task is not None

            action = self(task, state)
            instruction = [self.action_to_word(action, state.world)]
            return instruction

        goal_name, goal_arg = instruction
        task = self.task_map[goal_arg][goal_name]
        assert task is not None

        subtask = self.find_incomplete_subtask(task, state)
        world = state.world

        if subtask is None:
            return ["stop"]

        if subtask == task:
            action = self(task, state, debug=debug)
            instruction = [self.action_to_word(action, world)]
            return instruction

        return str(subtask).split()

    def get_task_from_item_name(self, item_name, arg_set=None):
        for task in self.task_manager.tasks:
            if task.goal_arg == item_name:
                return task
        return None

    def recognize_task(self, state_seq, action_seq):

        task_names = []

        if not self.config.teacher.primitive_language_only:

            # Check make and get tasks
            inventory_diff = state_seq[-1].inventory - state_seq[0].inventory
            last_inventory_diff = state_seq[-1].inventory - state_seq[-2].inventory

            added_one_item = inventory_diff[inventory_diff == 1].sum() == 1
            just_added_one_item = last_inventory_diff[last_inventory_diff == 1].sum() == 1

            if added_one_item and just_added_one_item:
                item_id = inventory_diff.tolist().index(1)
                item_name = state_seq[-1].get_item_name_by_id(item_id)
                task = self.task_map[item_name]['get']
                if task is not None:
                    task_names.append(str(task))
                else:
                    task = self.task_map[item_name]['make']
                    if task is not None:
                        task_names.append(str(task))

            # Check go tasks
            if not inventory_diff.any():
                neighbor_pos = state_seq[-1].neighbors()[0]
                item_name = state_seq[-1].get_item_name_at(neighbor_pos)
                if item_name is not None:
                    task = self.task_map[item_name]['go']
                    assert task is not None, item_name
                    task_names.append(str(task))

        # Check primitive actions
        if len(state_seq) == 2:
            assert len(action_seq) == 1
            action = action_seq[0]

            if action in self.student_action_map:
                task_names.append(self.student_action_map[action])
            else:
                start_pos = state_seq[0].pos
                end_pos = state_seq[-1].pos
                pos_diff = (end_pos[0] - start_pos[0],
                            end_pos[1] - start_pos[1])

                if pos_diff == (0, 0):

                    # If last action, check inventory change
                    start_inventory = state_seq[0].inventory
                    end_inventory = state_seq[-1].inventory

                    if (start_inventory != end_inventory).any():
                        self.student_action_map[action] = 'use'
                else:
                    if pos_diff == (0, -1):
                        self.student_action_map[action] = 'down'
                    elif pos_diff == (0, 1):
                        self.student_action_map[action] = 'up'
                    elif pos_diff == (-1, 0):
                        self.student_action_map[action] = 'left'
                    elif pos_diff == (1, 0):
                        self.student_action_map[action] = 'right'

                if action in self.student_action_map:
                    task_names.append(self.student_action_map[action])
                else:
                    unmapped_actions = [action for action in self.actions
                        if action not in self.student_action_map]
                    if len(unmapped_actions) == 1:
                        action_name = unmapped_actions[0]
                        self.student_action_map[action] = action_name
                        #task_names.append(action_name)
                    #else:
                        #task_names.append(self.random.choice(unmapped_actions))

        task_name = None
        for name in task_names:
            if task_name is None or len(name) < len(task_name):
                task_name = name

        if task_name is None:
            return None

        return task_name.split()

    def describe(self, state_seq, action_seq, action_prob_seq):

        assert len(state_seq) - 1 == len(action_seq)

        if not action_seq:
            return []
            """
            state_seq = state_seq + [state_seq[-1]]
            action_seq = [state_seq[-1].world.actions.STOP.index]
            return [(['stop'], (state_seq, action_seq))]
            """

        n = len(state_seq)
        f = [1e9] * n
        prev = [None] * n

        f[0] = 0

        for i in range(1, n):
            f[i] = 1e9
            for j in range(i):
                state_subseq = state_seq[j:i + 1]
                action_subseq = action_seq[j:i]
                task_name = self.recognize_task(state_subseq, action_subseq)
                if task_name is not None and f[j] + len(task_name) < f[i]:
                    f[i] = f[j] + len(task_name)
                    prev[i] = (j, task_name)

        if prev[n - 1] is None:
            return None

        dataset = []
        i = n - 1
        while i > 0:
            j, descr = prev[i]

            state_subseq = state_seq[j:i + 1]
            action_subseq = action_seq[j:i]
            action_prob_subseq = action_prob_seq[j:i]

            state_subseq.append(state_subseq[-1])
            action_subseq.append(state_subseq[-1].world.actions.STOP.index)
            action_prob_subseq.append(1)

            dataset.append((descr, (state_subseq, action_subseq, action_prob_subseq)))
            i = j

        return list(reversed(dataset))

    """
    def should_stop(self, instruction, state):
        return self.instruct(instruction, state) == ['stop']
    """





