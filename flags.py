import jsonargparse
import yaml
import numpy as np

from misc.util import Struct


def update_config(source, target):
    for k in source.keys():
        if isinstance(source[k], dict):
            if k not in target:
                target[k] = {}
            update_config(source[k], target[k])
        elif source[k] is not None:
            target[k] = source[k]


def make_config():

    parser = jsonargparse.ArgumentParser()

    parser.add_argument('-config_file', type=str)

    parser.add_argument('-seed', type=int)
    parser.add_argument('-name', type=str)
    parser.add_argument('-recipes', type=str)
    parser.add_argument('-device_id', type=int)
    parser.add_argument('-data_dir', type=str)
    parser.add_argument('-traj_file', type=str)


    parser.add_argument('-world.name', type=str)
    parser.add_argument('-world.config', type=str)

    parser.add_argument('-student.name', type=str)
    parser.add_argument('-student.uncertainty_threshold', type=float)
    parser.add_argument('-student.model.name', type=str)
    parser.add_argument('-student.model.hidden_size', type=int)
    parser.add_argument('-student.model.word_embed_size', type=int)
    parser.add_argument('-student.model.dropout_ratio', type=float)
    parser.add_argument('-student.model.learning_rate', type=float)
    parser.add_argument('-student.model.load_from', type=str)
    parser.add_argument('-student.model.num_layers', type=int)

    parser.add_argument('-teacher.name', type=str)

    parser.add_argument('-trainer.name', type=str)
    parser.add_argument('-trainer.use_curriculum', type=int)
    parser.add_argument('-trainer.hints', type=str)
    parser.add_argument('-trainer.max_timesteps', type=int)
    parser.add_argument('-trainer.log_every', type=int)
    parser.add_argument('-trainer.batch_size', type=int)
    parser.add_argument('-trainer.policy_mix.init_rate', type=float)
    parser.add_argument('-trainer.policy_mix.decay_every', type=int)

    parser.add_argument('-sanity_check_1', type=int, default=0)
    parser.add_argument('-sanity_check_2', type=int, default=0)

    flags = parser.parse_args()

    with open(flags.config_file) as f:
        config = yaml.safe_load(f)

    update_config(jsonargparse.namespace_to_dict(flags), config)

    config = Struct(**config)

    return config


if __name__ == '__main__':
    config = make_config()
    print(config)
