from .dataset import Dataset
from .task import TaskManager

def load(config):

    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = Dataset(config, split)

    return datasets

