from .dataset import Dataset
from .task import TaskManager

def load(config):

    task_manager = TaskManager(config)

    datasets = {}
    for split in ['train', 'dev', 'test']:
        datasets[split] = Dataset(config, split, task_manager)

    return datasets

