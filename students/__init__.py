from .imitation import ImitationStudent
from .language import LanguageStudent
from .rl import RLStudent


def load(config):
    cls_name = config.student.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
