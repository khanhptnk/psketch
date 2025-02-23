#from .reflex import ReflexModel
#from .attentive import AttentiveModel
#from .modular import ModularModel
#from .modular_ac import ModularACModel
#from .keyboard import KeyboardModel
from .lstm_seq2seq import LSTMSeq2SeqModel
from .transformer_seq2seq import TransformerSeq2SeqModel

def load(config):
    cls_name = config.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
