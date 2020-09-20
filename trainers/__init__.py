from .imitation import ImitationTrainer
from .describer import DescriberTrainer
from .executor import ExecutorTrainer
from .describer_pragmatic import DescriberPragmaticTrainer
from .language import LanguageTrainer
from .unsupervised import UnsupervisedTrainer
from .rl import RLTrainer


def load(config):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such trainer: {}".format(cls_name))
