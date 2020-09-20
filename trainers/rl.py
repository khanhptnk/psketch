import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import re
import editdistance
from termcolor import colored

import torch

from misc import util
from .imitation import ImitationTrainer


class RLTrainer(ImitationTrainer):

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

        src_words = []
        tasks = []
        regexes = []

        batch_size = len(batch)
        for item in batch:
            src_words.append(item['src_word'])
            tasks.append(item['task'])
            regexes.append(item['regex'])

        student.init(src_words, tasks, True)

        gold_tgt_words = teacher.predict(src_words, tasks)
        with torch.no_grad():
            pred_regexes = student.predict(src_words, tasks, sample=True)
        pred_tgt_words = self.execute_regex(src_words, pred_regexes)


        rewards = []
        for pred, gold in zip(pred_tgt_words, gold_tgt_words):
            if pred is not None:
                pred = ''.join(pred)
                gold = ''.join(gold)
                rewards.append(-editdistance.eval(pred, gold) / len(gold))
            else:
                rewards.append(-10)

        if pred_tgt_words[0] is not None:
            print(colored(''.join(gold_tgt_words[0]), 'red'),
                  colored(''.join(pred_tgt_words[0]), 'green'),
                  ''.join(pred_regexes[0]),
                  rewards[0])
        else:
            print('None', ''.join(pred_regexes[0]))



        student.receive(src_words, tasks, pred_regexes, rewards)

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
