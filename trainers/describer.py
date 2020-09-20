import logging
import os
import sys
sys.path.append('..')
import itertools
import json

import torch

from misc import util


class DescriberTrainer(object):

    def __init__(self, config):
        self.config = config

    def do_rollout(self, batch, describer, is_eval):

        words = []
        tasks = []

        batch_size = len(batch)
        for item in batch:
            examples.append([])
            for src_word, tgt_word in item['examples']:
                word = src_word + '@' + tgt_word
                examples[-1].append(word)
            tasks.append(item['task'])

        describer.init(examples, is_eval)

        t = 0
        golds = [None] * batch_size
        while not describer.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(tasks[i]):
                    golds[i] = tasks[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            describer.act(gold_actions=golds)
            t += 1

    def train(self, datasets, describer):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0
        best_eval_loss = 1e9

        for batch in itertools.cycle(datasets['train'].iterate_batches()):

            i_iter += 1

            self.do_rollout(batch, describer, False)

            loss = describer.learn()
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
                    describer.save('iter_%d' % i_iter)
                else:
                    describer.save('last')

                # Save best model
                eval_info = self.evaluate(datasets['val'], describer)
                eval_loss = eval_info['loss']
                eval_preds = eval_info['pred']
                if eval_loss < best_eval_loss:
                    logging.info('New best loss: %.1f' % eval_loss)
                    best_eval_loss = eval_loss
                    describer.save('best_dev')
                    self.save_preds('best_dev', eval_preds)

            if i_iter >= max_iters:
                break

    def evaluate(self, dataset, describer, save_pred=False):

        losses = []
        all_preds = {}

        for i, batch in enumerate(dataset.iterate_batches()):

            with torch.no_grad():

                # Compute loss on unseen data
                self.do_rollout(batch, describer, False)
                loss = describer.compute_loss().item()
                losses.append(loss)

                # Make predictions
                examples = []
                for item in batch:
                    examples.append([])
                    for src_word, tgt_word in item['examples']:
                        word = src_word + '@' + tgt_word
                        examples[-1].append(word)
                preds = describer.predict(examples)

            for item, pred in zip(batch, preds):
                assert item['id'] not in all_preds
                all_preds[item['id']] = { 'pred': ' '.join(pred)  }
                all_preds[item['id']].update(item)
                all_preds[item['id']]['src_word'] = ''.join(all_preds[item['id']]['src_word'])
                all_preds[item['id']]['tgt_word'] = ''.join(all_preds[item['id']]['tgt_word'])
                all_preds[item['id']]['task'] = ' '.join(all_preds[item['id']]['task'])

        for instance in dataset:
            assert instance['id'] in all_preds, instance['id']

        avg_loss = sum(losses) / len(losses)

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += 'loss = %.1f' % avg_loss
        logging.info(log_str)

        if save_pred:
            self.save_preds(dataset.split, all_preds)

        eval_info = {
                'loss': avg_loss,
                'pred': all_preds,
            }

        return eval_info

    def save_preds(self, filename, all_preds):
        file_path = os.path.join(self.config.experiment_dir, filename + '.pred')
        with open(file_path, 'w') as f:
            json.dump(all_preds, f, indent=2)
        logging.info('Saved eval info to %s' % file_path)

