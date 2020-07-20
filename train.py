import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

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

    with torch.cuda.device(config.device_id):
        trainer.train(datasets, world, student, teacher)

def configure():

    config = flags.make_config()

    config.command_line = 'python -u ' + ' '.join(sys.argv)

    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert not os.path.exists(config.experiment_dir), \
            "Experiment %s already exists!" % config.experiment_dir
    os.mkdir(config.experiment_dir)

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    config.device = torch.device('cuda', config.device_id)

    config.start_time = time.time()

    log_file = os.path.join(config.experiment_dir, 'run.log')
    util.config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info(config.command_line)
    logging.info('Write log to %s' % log_file)
    logging.info(str(config))

    return config

if __name__ == '__main__':
    main()
