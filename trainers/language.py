import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import re
from termcolor import colored

import torch

from misc import util
from .imitation import ImitationTrainer


class LanguageTrainer(ImitationTrainer):

    def __init__(self, config):
        self.config = config

    def execute_regex(self, src_words, regexes):

        tgt_words = []
        for i, regex in enumerate(regexes):
            regex = ''.join(regex)[1:-1]
            if '@' in regex and len(regex.split('@')) == 2:
                before, after = regex.split('@')
                if 'C' in after or 'V' in after or '(' in after or ')' in after:
                    tgt_words.append(None)
                    continue
                before = before.replace('C', '[^aeiou]').replace('V', '[aeiou]')
                after = '\\1' + after + '\\3'
                src_word = ''.join(src_words[i])[1:-1]
                try:
                    tgt_word = list('<' + re.sub(before, after, src_word) + '>')
                    tgt_words.append(tgt_word)
                except:
                    tgt_words.append(None)
            else:
                tgt_words.append(None)

        assert len(tgt_words) == len(src_words)

        return tgt_words

    def do_rollout(self, batch, student, teacher, is_eval):

        batch_size = len(batch)
        n_examples = len(batch[0]['examples'])

        src_words = []
        example_src_words = []
        example_tgt_words = []
        tasks = []

        for item in batch:
            for src_word, tgt_word in item['examples']:
                example_src_words.append(src_word)
                example_tgt_words.append(tgt_word)
            src_words.append(item['src_word'])
            tasks.append(item['task'])

        student.reset()

        space = '%30s'

        print(space % 'src_word:', example_src_words[0])
        print(space % 'tgt_word:', example_tgt_words[0])
        print(space % 'true regex:', colored(''.join(batch[0]['regex']), 'red'))
        print(space % 'task:', ' '.join(tasks[0]))

        # Decode the first time (with exploration) for language learning
        student.init('exploration', src_words, tasks, True)
        while not student.has_terminated():
            student.act(sample=True)
        pred_regexes = student.get_action_seqs()

        print(space % 'sample regex:', colored(''.join(pred_regexes[0]), 'cyan'))

        pred_example_regexes = []
        for regex in pred_regexes:
            pred_example_regexes.extend([regex] * n_examples)

        pred_example_tgt_words = self.execute_regex(
            example_src_words, pred_example_regexes)

        queried_examples = []
        queried_tgt_words = []
        queried_indices = []
        for i in range(batch_size):

            start_ix = i * n_examples
            end_ix = (i + 1) * n_examples

            all_equal = True
            tgt_none = False
            for j in range(start_ix, end_ix):
                if pred_example_tgt_words[j] is None:
                    tgt_none = True
                else:
                    src_word = example_src_words[j]
                    tgt_word = ''.join(pred_example_tgt_words[j])
                    if src_word != tgt_word:
                        all_equal = False

            if not all_equal and not tgt_none:
                queried_indices.append(i)
                queried_examples.append([])
                for j in range(start_ix, end_ix):
                    src_word = example_src_words[j]
                    tgt_word = ''.join(pred_example_tgt_words[j])
                    queried_examples[-1].append(src_word + '@' + tgt_word)

        if pred_example_tgt_words[0] is not None:
            print(space % 'sample tgt word:', ''.join(pred_example_tgt_words[0]))
        else:
            print(space % 'sample tgt word: None')

        descriptions = teacher.describe(queried_examples)

        if 0 in queried_indices and descriptions[0] is not None:
            print(space % 'sample task:', colored(' '.join(descriptions[0]), 'yellow'))
        else:
            print(space % 'sample task: None')

        added_src_words = []
        added_tasks = []
        added_regexes = []

        for i, ix in enumerate(queried_indices):
            if descriptions[i] is not None:
                start_ix = ix * n_examples
                end_ix = (ix + 1) * n_examples
                for j in range(start_ix, end_ix):
                    #print(example_src_words[j], descriptions[i], pred_regexes[ix])
                    added_src_words.append(example_src_words[j])
                    added_tasks.append(descriptions[i])
                    added_regexes.append(pred_regexes[ix])

        student.receive(added_src_words, added_tasks, added_regexes)

        # Decode the second time (without exploration) for task learning
        student.init('execution', src_words, tasks, True)
        while not student.has_terminated():
            student.act()
        pred_regexes = student.get_action_seqs()

        print(space % 'pred regex: ', colored(''.join(pred_regexes[0]), 'green'))

        pred_example_regexes = []
        for regex in pred_regexes:
            pred_example_regexes.extend([regex] * n_examples)

        pred_example_tgt_words = self.execute_regex(
            example_src_words, pred_example_regexes)

        if pred_example_tgt_words[0] is not None:
            print(space % 'pred tgt word:', ''.join(pred_example_tgt_words[0]))
        else:
            print(space % 'pred tgt word: None')

        added_src_words = []
        added_tasks = []
        added_regexes = []

        example_tasks = []
        for task in tasks:
            example_tasks.extend([task] * n_examples)

        is_correct = teacher.eval(
            example_src_words, example_tasks, pred_example_tgt_words)

        print(space % 'is correct: ', is_correct[0])

        for i in range(batch_size):
            start_ix = i * n_examples
            end_ix = (i + 1) * n_examples
            if all(is_correct[start_ix:end_ix]):
                print(colored(' '.join(tasks[i]), 'magenta'))
                for j in range(start_ix, end_ix):
                    added_src_words.append(example_src_words[j])
                    added_tasks.append(tasks[i])
                    added_regexes.append(pred_regexes[i])

        student.receive(added_src_words, added_tasks, added_regexes)

        print()

    def train(self, datasets, student, teacher):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0
        best_eval_score = -1e9

        for batch in itertools.cycle(datasets['train'].iterate_batches()):

            i_iter += 1

            self.do_rollout(batch, student, teacher, False)

            loss = student.learn()
            total_loss += loss

            if i_iter % log_every == 0:

                avg_loss = total_loss / log_every
                total_loss = 0

                log_str = 'Train iter %d (%d%%): ' % \
                    (i_iter, i_iter / max_iters * 100)
                log_str += 'loss = %.4f' % avg_loss

                logging.info('')
                logging.info(log_str)

                # Save last model
                if self.config.trainer.save_every_log:
                    student.save('iter_%d' % i_iter)
                else:
                    student.save('last')

                # Save best model
                eval_info = self.evaluate(datasets['val'], student)
                eval_score = eval_info['score']
                eval_preds = eval_info['pred']
                if eval_score > best_eval_score:
                    logging.info('New best score: %.1f' % eval_score)
                    best_eval_score = eval_score
                    student.save('best_dev')
                    self.save_preds('best_dev', eval_preds)
                self.save_preds('last', eval_preds)

            if i_iter >= max_iters:
                break



