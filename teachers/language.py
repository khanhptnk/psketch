import os
import sys
sys.path.append('..')
from termcolor import colored

import describers
import executors


class LanguageTeacher(object):

    def __init__(self, config):

        self.describer = describers.load(config)
        self.executor  = executors.load(config)

    def describe(self, examples):

        if not examples:
            return []

        pred_tasks, pred_tgt_words, scores = self.describer.pragmatic_predict(
            examples, self.executor)

        n_examples = len(examples[0])
        for i, score in enumerate(scores):
            if score < n_examples:
                pred_tasks[i] = None
            else:
                print(colored(' '.join(pred_tasks[i]) + '\n' + str(examples[i]), 'yellow'))

        return pred_tasks

    def eval(self, src_words, tasks, pred_tgt_words):
        tgt_words = self.executor.predict(src_words, tasks)
        if tgt_words[0] == pred_tgt_words[0]:
            print(tasks[0], ''.join(tgt_words[0]), ''.join(pred_tgt_words[0]))
        return [pred == gold for pred, gold in zip(pred_tgt_words, tgt_words)]

    def predict(self, src_words, tasks):
        return self.executor.predict(src_words, tasks)



