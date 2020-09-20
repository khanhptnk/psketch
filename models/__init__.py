#from .reflex import ReflexModel
#from .attentive import AttentiveModel
#from .modular import ModularModel
#from .modular_ac import ModularACModel
#from .keyboard import KeyboardModel
from .lstm_seq2seq import LSTMSeq2SeqModel
from .transformer_seq2seq import TransformerSeq2SeqModel
from .describer_lstm_seq2seq import DescriberLSTMSeq2SeqModel
from .executor_lstm_seq2seq import ExecutorLSTMSeq2SeqModel
from .student_lstm_seq2seq import StudentLSTMSeq2SeqModel
from .student_rl_lstm_seq2seq import StudentRLLSTMSeq2SeqModel

def load(config):
    cls_name = config.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
