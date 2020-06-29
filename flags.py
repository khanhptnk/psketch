import jsonargparse
import yaml

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

    parser.add_argument('-config_file', type=str, default='configs/config.yaml')

    parser.add_argument('-name', type=str)
    parser.add_argument('-recipes', type=str)

    parser.add_argument('-world.name', type=str)

    parser.add_argument('-model.name', type=str)
    parser.add_argument('-model.use_args', type=int)
    parser.add_argument('-model.featurize_plan', type=int)
    parser.add_argument('-model.max_sub_task_timesteps', type=int)
    parser.add_argument('-model.baseline', type=str)

    parser.add_argument('-trainer.name', type=str)
    parser.add_argument('-trainer.use_curriculum', type=int)
    parser.add_argument('-trainer.improvement_threshold', type=float)
    parser.add_argument('-trainer.hints', type=str)
    parser.add_argument('-trainer.max_timesteps', type=int)

    flags = parser.parse_args()

    with open(flags.config_file) as f:
        config = yaml.safe_load(f)

    update_config(jsonargparse.namespace_to_dict(flags), config)

    return Struct(**config)


if __name__ == '__main__':
    config = make_config()
    print(config)
