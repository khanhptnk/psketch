import os
import sys
from collections import defaultdict

from .base import BaseTeacher
from .demonstration import DemonstrationTeacher
from .primitive_language import PrimitiveLanguageTeacher


class InteractivePrimitiveLanguageTeacher(PrimitiveLanguageTeacher):

    def __init__(self, config):

        super(InteractivePrimitiveLanguageTeacher, self).__init__(config)
        self.student_action_map = {}
        self.random = config.random
        self.demonstration_teacher = DemonstrationTeacher(config)

    def __call__(self, task, state):
        world = state.world
        action = self.demonstration_teacher(task, state)
        instruction = [self.action_to_word(action, world)]
        return instruction

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






