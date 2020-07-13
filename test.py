import os
import sys

import worlds
import teachers
import models
import trainers
import flags
import data


config = flags.make_config()
world = worlds.load(config)
teacher = teachers.load(config)
datasets = data.load(config)

example = datasets['train'][0]
task_manager = data.TaskManager(config)

task = task_manager['make[shears]']
print(task)

scenario = world.make_scenario(example['grid'], example['init_pos'])

state = scenario.init()

state.render()

ref_action = teacher(task, state)

while ref_action != -1:
    _, state = state.step(ref_action)
    state.render()
    ref_action = teacher(task, state)
