import os
import sys
from collections import defaultdict

from .base import BaseTeacher
from .demonstration import DemonstrationTeacher


class PrimitiveLanguageTeacher(DemonstrationTeacher):

    def __init__(self, config):

        super(PrimitiveLanguageTeacher, self).__init__(config)
        self.student_action_map = {}
        self.random = config.random

    def instruct(self, world, action_seq):
        instruction = []
        for action in action_seq:
            if action == world.actions.UP.index:
                instruction.append('up')
            elif action == world.actions.DOWN.index:
                instruction.append('down')
            elif action == world.actions.LEFT.index:
                instruction.append('left')
            elif action == world.actions.RIGHT.index:
                instruction.append('right')
            elif action == world.actions.USE.index:
                instruction.append('use')
            else:
                assert action == world.actions.STOP.index
                instruction.append('stop')
        return instruction

    def describe(self, world, action_seq, state_seq):

        description = []

        for i, action in enumerate(action_seq):
            action_str = None
            # Check if teacher has stored mapping from action index to string
            if action in self.student_action_map:
                action_str = self.student_action_map[action]

            # Infer the last action
            if action_str is None and \
               len(self.student_action_map) == len(world.action_space) - 1:
                recognized_actions = list(self.student_action_map.values())
                for w in ['up', 'down', 'left', 'right', 'use', 'stop']:
                    if w not in recognized_actions:
                        self.student_action_map[action] = w
                        action_str = w
                        break

            if action_str is None:
                this_pos = state_seq[i + 1].pos
                prev_pos = state_seq[i].pos
                diff = (this_pos[0] - prev_pos[0], this_pos[1] - prev_pos[1])

                if diff == (0, 0):

                    # If last action, check inventory change
                    this_inventory = state_seq[i + 1].inventory
                    prev_inventory = state_seq[i].inventory

                    if (this_inventory != prev_inventory).any():
                        self.student_action_map[action] = 'use'
                        action_str = 'use'
                    else:
                        candidates = ['down', 'up', 'left', 'right', 'use']
                        if i + 1 == len(state_seq) - 1:
                            candidates.append('stop')
                        action_str = self.random.choice(candidates)
                else:
                    if diff == (0, -1):
                        self.student_action_map[action] = 'down'
                    elif diff == (0, 1):
                        self.student_action_map[action] = 'up'
                    elif diff == (-1, 0):
                        self.student_action_map[action] = 'left'
                    elif diff == (1, 0):
                        self.student_action_map[action] = 'right'
                    action_str = self.student_action_map[action]

            assert action_str is not None
            description.append(action_str)

        return description









