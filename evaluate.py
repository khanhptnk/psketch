import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch

import flags
import worlds
import data
import trainers
import students
import teachers

from misc import util


def main():

    config = configure()
    world = worlds.load(config)
    datasets = data.load(config)
    trainer = trainers.load(config)
    student = students.load(config)
    teacher = teachers.load(config)

    logging.info(str(student.model))

    with torch.cuda.device(config.device_id):
        _, dev_eval_info = trainer.evaluate(
            datasets['dev'], world, student, teacher, save_traj=True)
        breakdown_results(dev_eval_info, datasets['dev'])
        _, test_eval_info = trainer.evaluate(
            datasets['test'], world, student, teacher, save_traj=True)
        breakdown_results(test_eval_info, datasets['test'])

def breakdown_results(eval_info, dataset):
    success_table = defaultdict(list)
    for instance in dataset:
        instance_id = instance['id']
        task = instance['task']
        success_table[task.goal_name].append(eval_info[instance_id]['success'])
        success_table[str(task)].append(eval_info[instance_id]['success'])

    for k, v in success_table.items():
        logging.info('%15s %.1f' % (k, sum(v) / len(v) * 100))

def configure():

    config = flags.make_config()

    config.command_line = 'python -u ' + ' '.join(sys.argv)

    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert os.path.exists(config.experiment_dir), \
            "Experiment %s not exists!" % config.experiment_dir

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    config.device = torch.device('cuda', config.device_id)

    config.start_time = time.time()

    log_file = os.path.join(config.experiment_dir, 'run.log')
    util.config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info(config.command_line)
    logging.info(str(config))

    return config

if __name__ == '__main__':
    main()
