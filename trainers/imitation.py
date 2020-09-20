import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import re

import torch

from misc import util


class ImitationTrainer(object):

    def __init__(self, config):
        self.config = config

    def do_rollout(self, batch, student, is_eval):

        src_words = []
        tasks = []
        regexes = []

        batch_size = len(batch)
        for item in batch:
            src_words.append(item['src_word'])
            tasks.append(item['task'])
            regexes.append(item['regex'])

        student.init(src_words, tasks, is_eval)

        t = 0
        golds = [None] * batch_size
        while not student.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(regexes[i]):
                    golds[i] = regexes[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            student.act(gold_actions=golds)
            t += 1

    def train(self, datasets, student):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0
        best_eval_score = -1e9

        for batch in itertools.cycle(datasets['train'].iterate_batches()):

            i_iter += 1

            self.do_rollout(batch, student, False)

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

    def evaluate(self, dataset, student, save_pred=False):

        all_scores = []
        all_preds = []

        for i, batch in enumerate(dataset.iterate_batches()):

            with torch.no_grad():

                # Make predictions
                src_words = [item['src_word'] for item in batch]
                tasks = [item['task'] for item in batch]
                preds = student.predict(src_words, tasks)

                scores = [None] * len(batch)
                for i, regex in enumerate(preds):
                    src_word = ''.join(batch[i]['src_word'])[1:-1]
                    tgt_word = ''.join(batch[i]['tgt_word'])[1:-1]
                    regex = ''.join(regex)[1:-1]
                    if '@' not in regex or len(regex.split('@')) != 2:
                        scores[i] = 0
                    else:
                        before, after = regex.split('@')
                        before = before.replace('C', '[^aeiou]').replace('V', '[aeiou]')
                        after = '\\1' + after + '\\3'
                        try:
                            pred_tgt_word = re.sub(before, after, src_word)
                            scores[i] = pred_tgt_word == tgt_word
                        except:
                            scores[i] = 0
                all_scores.extend(scores)

            for item, pred in zip(batch, preds):
                new_item = { 'pred': ''.join(pred)  }
                new_item.update(item)
                new_item['src_word'] = ''.join(new_item['src_word'])
                new_item['tgt_word'] = ''.join(new_item['tgt_word'])
                new_item['task'] = ' '.join(new_item['task'])
                new_item['regex'] = ''.join(new_item['regex'])
                all_preds.append(new_item)

        score = sum(all_scores) / len(all_scores) * 100

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += 'score = %.1f' % score
        logging.info(log_str)

        if save_pred:
            self.save_preds(dataset.split, all_preds)

        eval_info = {
                'score': score,
                'pred': all_preds,
            }

        return eval_info

    def save_preds(self, filename, all_preds):
        file_path = os.path.join(self.config.experiment_dir, filename + '.pred')
        with open(file_path, 'w') as f:
            json.dump(all_preds, f, indent=2)
        logging.info('Saved eval info to %s' % file_path)

