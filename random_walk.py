import os
import sys
import yaml
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from collections import defaultdict, namedtuple

import flags
import worlds
import models

from misc.experience import Transition
from worlds.cookbook import Cookbook
from misc import util


#Task = namedtuple("Task", ["goal", "steps"])

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


class TaskManager(object):

    def __init__(self, config):
        # load configs
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.subtask_index = util.Index()
        self.task_index = util.Index()
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.safe_load(hints_f)

        # initialize randomness
        self.random = np.random.RandomState(0)

        # organize task and subtask indices
        self.tasks_by_subtask = defaultdict(list)
        self.tasks = []
        for hint_key, hint in self.hints.items():
            goal = util.parse_fexp(hint_key)
            goal = (self.subtask_index.index(goal[0]), self.cookbook.index[goal[1]])
            if config.model.use_args:
                steps = [util.parse_fexp(s) for s in hint]
                steps = [(self.subtask_index.index(a), self.cookbook.index[b])
                        for a, b in steps]
                steps = tuple(steps)
                task = Task(goal, steps)
                for subtask, _ in steps:
                    self.tasks_by_subtask[subtask].append(task)
            else:
                steps = [self.subtask_index.index(a) for a in hint]
                steps = tuple(steps)
                task = Task(goal, steps)
                for subtask in steps:
                    self.tasks_by_subtask[subtask].append(task)
            self.tasks.append(task)
            self.task_index.index(task)


#random.seed(12321)

config = flags.make_config()

task_manager = TaskManager(config)
world = worlds.load(config)
model = models.load(config)


model.prepare(world, None)

N_BATCH = 1

states_before = []
tasks = []
goal_names = []
goal_args = []

for _ in range(N_BATCH):
    task = random.choice(task_manager.tasks)
    goal, _ = task
    goal_name, goal_arg = goal
    scenario = world.sample_scenario_with_goal(goal_arg)
    states_before.append(scenario.init())

    states_before[-1].world.render(states_before[-1])

    tasks.append(task)
    goal_names.append(goal_name)
    goal_args.append(goal_arg)

model.init(states_before, tasks)
transitions = [[] for _ in range(N_BATCH)]

T = 1000
timer = T
done = [False for _ in range(N_BATCH)]

while not all(done) and timer > 0:

    mstates_before = model.get_state()
    action, terminate = model.act(states_before)
    mstates_after = model.get_state()
    states_after = [None] * N_BATCH

    for i in range(N_BATCH):
        if action[i] is None:
            assert done[i]
        elif terminate[i]:
            win = states_before[i].satisfies(goal_names[i], goal_args[i])
            reward = 1 if win else 0
        elif action[i] > world.n_actions:
            states_afer[i] = states_before[i]
            reward = 0
        else:
            reward, states_after[i] = states_before[i].step(action[i])

        if not done[i]:
            transitions[i].append(Transition(
                states_before[i], mstates_before[i], action[i],
                states_after[i], mstates_after[i], reward))

        states_after[i].world.render(states_after[i])

        if terminate[i]:
            done[i] = True

    states_before = states_after
    timer -= 1







