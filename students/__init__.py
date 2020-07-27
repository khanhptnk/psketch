#from .reflex import ReflexModel
#from .attentive import AttentiveModel
#from .modular import ModularModel
#from .modular_ac import ModularACModel
#from .keyboard import KeyboardModel
from .imitation import ImitationStudent
from .primitive_language import PrimitiveLanguageStudent
from .interactive_primitive_language import InteractivePrimitiveLanguageStudent


def load(config):
    cls_name = config.student.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
