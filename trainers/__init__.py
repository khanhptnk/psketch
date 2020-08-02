from .imitation import ImitationTrainer
from .primitive_language import PrimitiveLanguageTrainer
from .interactive_primitive_language import InteractivePrimitiveLanguageTrainer
from .active_primitive_language import ActivePrimitiveLanguageTrainer

def load(config):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such trainer: {}".format(cls_name))
