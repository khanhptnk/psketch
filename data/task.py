import os
import sys
sys.path.append('..')
import yaml

from misc import util


class Task(object):

    def __init__(self, goal, subtasks=None):
        self.goal_name, self.goal_arg = util.parse_fexp(goal)
        self.subtasks = None
        if subtasks:
            self.subtasks = subtasks
        self.encoding = None

    def __repr__(self):
        return f"Task({self.goal_name}[{self.goal_arg}])"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.goal_name == other.goal_name and \
               self.goal_arg  == other.goal_arg

    def __str__(self):
        return self.goal_name + ' ' + self.goal_arg


class TaskManager(object):

    def __init__(self, config):
        # load configs
        self.config = config
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.safe_load(hints_f)

        # organize task and subtask indices
        self.tasks_by_goal = {}
        self.tasks = util.Index()
        for goal, subgoals in self.hints.items():
            subtasks = [self.tasks_by_goal[subgoal] for subgoal in subgoals]
            task = Task(goal, subtasks)
            self.tasks_by_goal[goal] = task
            self.tasks.index(task)

        # make vocab
        self.vocab = util.Index()
        self.vocab.index('<EOS>')
        self.vocab.index('<PAD>')
        for task in self.tasks:
            self.vocab.index(task.goal_name)
            if task.goal_arg:
                self.vocab.index(task.goal_arg)

        for task in self.tasks:
            task.encoding = self.encode_task(task)

        #config.student.model.pad_idx = self.vocab['<PAD>']
        #config.student.model.vocab_size = len(self.vocab)
        config.vocab = self.vocab

    def encode_task(self, task):
        return [self.vocab[task.goal_name], self.vocab[task.goal_arg]]

    def decode_task(self, indices):
        goal_name = self.vocab.get(indices[0])
        goal_arg = self.vocab.get(indices[0])
        goal = goal_name + '[' + goal_arg + ']'
        return self.tasks[goal]

    def __getitem__(self, goal):
        return self.tasks_by_goal[goal]
