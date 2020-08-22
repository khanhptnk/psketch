import os
import sys

from .base import BaseTeacher


class DemonstrationTeacher(BaseTeacher):

    def __call__(self, task, state, debug=False):

        actions = state.world.actions

        if isinstance(task, list):
            task = ' '.join(task)
            if task == 'left':
                return actions.LEFT.index
            elif task == 'right':
                return actions.RIGHT.index
            elif task == 'up':
                return actions.UP.index
            elif task == 'down':
                return actions.DOWN.index
            elif task == 'use':
                return actions.USE.index
            elif task == 'stop':
                return actions.STOP.index
            else:
                task = self.task_manager[task]

        subtask = self.find_incomplete_subtask(task, state)

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



