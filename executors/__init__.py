from .executor import Executor


def load(config):
    cls_name = config.executor.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
