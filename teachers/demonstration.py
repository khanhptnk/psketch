import os
import sys

from .base import BaseTeacher


class DemonstrationTeacher(BaseTeacher):

    def __call__(self, task, state):

        subtask = self.find_incomplete_subtask(task, state)

        actions = state.world.actions

        if subtask is None:
            return actions.STOP.index

        assert subtask.goal_name in ['use', 'go']

        if subtask.goal_name == 'use':
            return actions.USE.index

        _, best_action_seq = self.find_closest_resources(subtask, state)

        if best_action_seq is None:
            return actions.STOP.index

        ref_action = best_action_seq[0]

        return ref_action



