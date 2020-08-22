import os
import sys
import json
import logging
import numpy as np
sys.path.append('..')

from .task import TaskManager
from misc import util

class Dataset(object):

    def __init__(self, config, split, task_manager):

        self.config = config
        self.split = split
        self.task_manager = task_manager
        self.file_name = os.path.join(
            config.data_dir, config.world.config + '_' + split + '.json')
        self.data = self.load_data(self.file_name)
        self.instance_by_id = {}
        for item in self.data:
            self.instance_by_id[item['id']] = item
        self.item_idx = 0
        self.random = config.random
        self.batch_size = config.trainer.batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def get_instance_by_id(self, instance_id):
        return self.instance_by_id[instance_id]

    def load_data(self, file_name):
        with open(file_name) as f:
            data = json.load(f)
        data = self.flatten_data(data)
        logging.info('Loaded %d instances of %s split from %s' %
            (len(data), self.split, file_name))
        return data

    def flatten_data(self, data):
        new_data = []
        for item in data:
            grid = item['grid']
            for task_instance in item['task_instances']:
                task_name = ' '.join(util.parse_fexp(task_instance['task']))
                task = self.task_manager[task_name]
                init_positions = task_instance['init_pos']
                ids = task_instance['ids']
                ref_actions_seqs = task_instance['ref_actions']
                zipped_info = zip(init_positions, ids, ref_actions_seqs)
                for pos, id, ref_actions in zipped_info:
                    new_item = {
                        'id'          : id,
                        'task'        : task,
                        'grid'        : np.array(grid),
                        'init_pos'    : tuple(pos),
                        'ref_actions' : tuple(ref_actions)
                    }
                    new_data.append(new_item)
        return new_data

    def next_batch(self):
        if self.item_idx == 0:
            self.data_indices = list(range(len(self)))
            self.random.shuffle(self.data_indices)

        start_idx = self.item_idx
        end_idx = self.item_idx + self.batch_size
        batch_indices = self.data_indices[start_idx:end_idx]
        self.item_idx = end_idx

        end_pass = False
        if self.item_idx >= len(self):
            self.item_idx = 0
            end_pass = True

        batch = [self[idx] for idx in batch_indices]

        return batch, end_pass

    def iterate_batches(self):
        end_pass = False
        while not end_pass:
            batch, end_pass = self.next_batch()
            yield batch

