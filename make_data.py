import os
import sys
import yaml
import json
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from collections import defaultdict, namedtuple

import flags
import worlds
import models
import data

from misc.experience import Transition
from worlds.cookbook import Cookbook
from misc import util
from worlds.craft import random_free, CraftScenario


WIDTH = 10
HEIGHT = 10

N_WORKSHOPS = 3


def is_connected(grid, init_pos):

    prev = {}
    queue = [None] * 1000
    start = 0
    end = 0
    queue[end] = init_pos
    end += 1
    prev[init_pos] = -1

    while start < end:
        pos = queue[start]
        start += 1

        for i, action in enumerate(world.action_space):

            if action in [world.actions.USE, world.actions.STOP]:
                continue

            coord_change = action.coord_change
            new_pos = (pos[0] + coord_change[0], pos[1] + coord_change[1])

            # only change direction if new position is not enterable
            if grid[new_pos[0], new_pos[1]]:
                new_pos = pos

            if new_pos not in prev:
                queue[end] = new_pos
                end += 1
                prev[new_pos] = -1

    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            if c == 0 and (i, j) not in prev:
                return False

    return True

def random_free(grid, random, keep_connected=True):
    nav_grid = grid.max(axis=2)
    pos = None
    while pos is None:
        (x, y) = (random.randint(WIDTH), random.randint(HEIGHT))
        if nav_grid[x, y]:
            continue
        is_good = True
        if keep_connected:
            nav_grid[x, y] = 1
            for i, row in enumerate(nav_grid):
                for j, c in enumerate(row):
                    if c == 1 and 0 < i < WIDTH - 1 and 0 < j < HEIGHT - 1 and\
                    not is_connected(nav_grid, (i, j)):
                        is_good = False
                        break
        if is_good:
            pos = (x, y)
        else:
            nav_grid[x, y] = 0
    return pos

def sample_scenario(world, ingredients, config, make_island=False, make_cave=False):
    # generate grid
    grid = np.zeros((WIDTH, HEIGHT, world.cookbook.n_kinds))
    i_bd = world.cookbook.index["boundary"]
    grid[0, :, i_bd] = 1
    grid[WIDTH-1:, :, i_bd] = 1
    grid[:, 0, i_bd] = 1
    grid[:, HEIGHT-1:, i_bd] = 1

    # treasure
    if make_island or make_cave:
        (gx, gy) = (1 + np.random.randint(WIDTH-2), 1)
        treasure_index = \
                world.cookbook.index["gold"] if make_island else world.cookbook.index["gem"]
        wall_index = \
                self.water_index if make_island else self.stone_index
        grid[gx, gy, treasure_index] = 1
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not grid[gx+i, gy+j, :].any():
                    grid[gx+i, gy+j, wall_index] = 1

    # ingredients
    for primitive in world.cookbook.primitives:
        if primitive == world.cookbook.index["gold"] or \
                primitive == world.cookbook.index["gem"]:
            continue
        for i in range(4):
            (x, y) = random_free(grid, config.random)
            grid[x, y, primitive] = 1

    # generate crafting stations
    for i_ws in range(N_WORKSHOPS):
        ws_x, ws_y = random_free(grid, config.random)
        grid[ws_x, ws_y, world.cookbook.index["workshop%d" % i_ws]] = 1

    # generate init pos
    init_pos = random_free(grid, config.random)

    return grid, init_pos

config = flags.make_config()
config.random = np.random.RandomState(123)
task_manager = data.TaskManager(config)
world = worlds.load(config)

ingredients = []
for i in ["wood", "grass", "iron"]:
    ingredients.append(world.cookbook.index[i])

seed_scenarios = []

N_WORLDS = 100
for i in range(N_WORLDS):
    print(i)
    while True:
        grid, init_pos = sample_scenario(world, ingredients, config)
        scenario = world.make_scenario(grid, init_pos)
        duplicate = False
        for prev_scenario in seed_scenarios:
            if (scenario.init_grid == prev_scenario.init_grid).all():
                duplicate = True
        if not duplicate:
            break
    scenario.init().render()
    seed_scenarios.append(scenario)

assert len(seed_scenarios) == N_WORLDS

N_POS = 20

data_by_env = []

i_instance = 0

for scenario in seed_scenarios:
    grid = scenario.init_grid
    item = {}
    item['grid'] = grid.tolist()
    item['task_instances'] = []
    for task in task_manager.tasks:

        if task.goal_name not in ['get', 'make']:
            continue

        task_instance = {}
        task_instance['task'] = str(task)
        task_instance['init_pos'] = []
        task_instance['ids'] = []
        while len(task_instance['init_pos']) < N_POS:
            pos = random_free(grid, config.random, keep_connected=False)
            if pos not in task_instance['init_pos']:
                i_instance += 1
                task_instance['ids'].append('instance_%d' % i_instance)
                task_instance['init_pos'].append(pos)
        assert len(task_instance['init_pos']) == N_POS
        item['task_instances'].append(task_instance)
    data_by_env.append(item)

config.random.shuffle(data_by_env)

datasets = {}

print(len(data_by_env))

N_TRAIN_ENV = 80
N_DEV_ENV = 10
N_DEV_ENV = 10

datasets['train'] = data_by_env[:N_TRAIN_ENV]
datasets['dev']   = data_by_env[N_TRAIN_ENV:(N_TRAIN_ENV + N_DEV_ENV)]
datasets['test']  = data_by_env[(N_TRAIN_ENV + N_DEV_ENV):]

"""
grid = np.array(datasets['train'][0]['scenario'])
task_instance = datasets['train'][0]['task_instances'][0]

for pos in task_instance['init_pos']:
    pos = tuple(pos)
    scenario = CraftScenario(grid, pos, world)
    state = scenario.init()
    print(state.pos)
    print(task_instance['task'])
    state.render()
    _, state = state.step(2)
    state.render()
    _, state = state.step(4)
    state.render()
    break
"""

for name in ['train', 'dev', 'test']:
    with open('data/' + name + '.json', 'w') as f:
        json.dump(datasets[name], f, indent=2)






