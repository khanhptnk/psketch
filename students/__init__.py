from .imitation import ImitationStudent
from .primitive_language import PrimitiveLanguageStudent
from .interactive_primitive_language import InteractivePrimitiveLanguageStudent
from .active_primitive_language import ActivePrimitiveLanguageStudent
from .abstract_language import AbstractLanguageStudent


def load(config):
    cls_name = config.student.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
