import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch

import flags
import data
import trainers
import executors

from misc import util


def main():

    config = configure()
    datasets = data.load(config)
    trainer = trainers.load(config)
    executor = executors.load(config)

    with torch.cuda.device(config.device_id):
        trainer.evaluate(datasets['val'], executor, save_pred=True)
        trainer.evaluate(datasets['test'], executor, save_pred=True)

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

    log_file = os.path.join(config.experiment_dir, 'eval.log')
    util.config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info(config.command_line)
    logging.info(str(config))

    return config

if __name__ == '__main__':
    main()
