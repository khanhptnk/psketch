import os
import sys
import json

import flags
import worlds
import data

import torch
import numpy as np


def configure():

    config = flags.make_config()

    config.command_line = 'python -u ' + ' '.join(sys.argv)

    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert os.path.exists(config.experiment_dir), \
            "Experiment %s not already exists!" % config.experiment_dir

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    return config

config = configure()
datasets = data.load(config)
world = worlds.load(config)

with open(config.traj_file) as f:
    eval_info = json.load(f)

if 'dev' in config.traj_file:
    split = 'dev'
elif 'test' in config.traj_file:
    split = 'test'
else:
    print('Please enter a different traj_file!')
    sys.exit(1)

dataset = datasets[split]

traj_ids = list(eval_info.keys())

while True:
    print('>> ', end='')
    instance_id = 'instance_' + input('Enter an instance ID: ')

    if instance_id == 'instance_rand':
        instance_id = config.random.choice(list(eval_info.keys()))
        print(instance_id)

    if instance_id in eval_info:
        result = eval_info[instance_id]
        instance = dataset.get_instance_by_id(instance_id)
        state = world.init_state(instance['grid'], instance['init_pos'])
        print(instance['task'], '(success)' if result['success'] else '(fail)')
        state.render()
        for a in result['actions']:
            key = input()
            if a == world.actions.STOP.index:
                break
            print(instance['task'], 'Action %d' % a)
            _, state = state.step(a)
            state.render()
            if key == 'q':
                break


