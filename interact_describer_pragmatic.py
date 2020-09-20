import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

import torch

import flags
import data
import trainers
import describers
import executors

from misc import util


def main():

    config = configure()
    datasets = data.load(config)
    describer = describers.load(config)
    executor = executors.load(config)

    with torch.no_grad():
        while True:
            print('Enter src word: ', end='')
            src_word = list('<' + input() + '>')
            print('Enter tgt word: ', end='')
            tgt_word = list('<' + input() + '>')
            preds, pred_tgts = describer.pragmatic_predict(
                [src_word], [tgt_word], executor)
            pred = preds[0]
            pred_tgt = pred_tgts[0]
            print('>> Instruction: ', ' '.join(pred))
            print('>> Pred tgt: ', ''.join(pred_tgt))
            print()


def configure():

    config = flags.make_config()

    config.command_line = 'python -u ' + ' '.join(sys.argv)

    config.data_dir = os.getenv('PT_DATA_DIR', 'data')
    output_dir = os.getenv('PT_OUTPUT_DIR', 'experiments')
    config.experiment_dir = os.path.join(output_dir + ("/%s" % config.name))

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
