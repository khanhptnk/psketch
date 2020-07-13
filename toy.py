from misc import util

class Task(object):

    def __init__(self, name, steps=None):
        self.goal_name, self.goal_arg = util.parse_fexp(name)
        self.subtasks = None
        if steps is not None:
            self.subtasks = [Task(subtask) for subtask in steps]

    def __repr__(self):
        return f"Task({self.goal_name}[{self.goal_arg}])"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.goal_name == other.goal_name and \
               self.goal_arg  == other.goal_arg


