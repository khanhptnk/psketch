import os
import sys
import time
import numpy as np

import torch

import flags
import worlds
import data
import trainers
import students
import teachers

def main():

    config = configure()
    world = worlds.load(config)
    datasets = data.load(config)
    trainer = trainers.load(config)
    student = students.load(config)
    teacher = teachers.load(config)

    print(config)
    print(student.model)

    with torch.cuda.device(config.device_id):
        trainer.train(datasets, world, student, teacher)

def configure():

    config = flags.make_config()
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert not os.path.exists(config.experiment_dir), \
            "Experiment %s already exists!" % config.experiment_dir
    os.mkdir(config.experiment_dir)

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    config.device = torch.device('cuda', config.device_id)

    config.start_time = time.time()

    return config

if __name__ == '__main__':
    main()
