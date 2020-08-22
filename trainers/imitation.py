import logging
import os
import sys
sys.path.append('..')
import itertools
import json

import torch

from misc import util


class ImitationTrainer(object):

    def __init__(self, config):
        self.config = config

    def do_rollout(self, batch, world, student, teacher, is_eval):
        states = []
        tasks = []
        goal_names, goal_args = [], []

        batch_size = len(batch)
        for item in batch:
            tasks.append(item['task'])
            states.append(world.init_state(item['grid'], item['init_pos']))

        student.init(tasks, states, is_eval)

        timer = [self.config.trainer.max_timesteps] * batch_size

        done = [False] * batch_size
        success = [False] * batch_size
        action_seqs = [[] for i in range(batch_size)]
        num_interactions = 0
        num_steps = 0

        if not is_eval:
            behavior_clone = self.config.random.binomial(
                1, self.policy_mix_rate, size=batch_size)

        while not all(done):
            actions = student.act(states)

            ref_actions = [None] * batch_size

            for i in range(batch_size):
                # Ask teacher for reference action
                if not is_eval:
                    if done[i]:
                        ref_actions[i] = -1
                    else:
                        ref_actions[i] = teacher(tasks[i], states[i])
                        num_interactions += (not is_eval and not done[i])

                    if behavior_clone[i]:
                        actions[i] = ref_actions[i]

                if not done[i]:
                    # Save action
                    action_seqs[i].append(actions[i])

                timer[i] -= 1
                done[i] |= actions[i] == world.actions.STOP.index or \
                           timer[i] <= 0

                # Transition to next state
                if done[i]:
                    success[i] = states[i].satisfies(tasks[i])
                    assert success[i] is not None
                else:
                    _, states[i] = states[i].step(actions[i])
                    num_steps += (not is_eval)

            # Receive reference actions
            if not is_eval:
                student.receive(ref_actions)

        distances = []
        for i in range(batch_size):
            if not done[i]:
                success[i] = states[i].satisfies(tasks[i])
            if tasks[i].goal_name == 'get':
                if not success[i]:
                    state = world.init_state(
                        batch[i]['grid'], states[i].pos, states[i].dir)
                    _, best_action_seq = teacher.find_closest_resources(
                        tasks[i], state)
                    distances.append(len(best_action_seq))
                else:
                    distances.append(0)

        info = {
                'action_seqs': action_seqs,
                'success': success,
                'distances': distances,
                'num_interactions': num_interactions,
                'num_steps': num_steps
            }

        return info

    def train(self, datasets, world, student, teacher):

        student.prepare(world)

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0
        total_success = (0, 0)
        total_distance = (0, 0)
        total_interactions = 0
        total_steps = 0
        best_eval_success_rate = -1e9

        self.policy_mix_rate = self.config.trainer.policy_mix.init_rate
        decay_every = self.config.trainer.policy_mix.decay_every

        for batch in itertools.cycle(datasets['train'].iterate_batches()):

            i_iter += 1

            info = self.do_rollout(batch, world, student, teacher, False)
            success = info['success']
            distances = info['distances']
            num_interactions = info['num_interactions']
            num_steps = info['num_steps']

            total_success = util.add_stat(total_success, success)
            total_distance = util.add_stat(total_distance, distances)
            total_interactions += num_interactions
            total_steps += num_steps

            loss = student.learn()
            total_loss += loss

            if i_iter % log_every == 0:

                avg_loss = total_loss / log_every
                avg_success_rate = total_success[0] / total_success[1] * 100
                avg_distance = total_distance[0] / total_distance[1]

                total_loss = 0

                log_str = 'Train iter %d (%d%%): ' % \
                    (i_iter, i_iter / max_iters * 100)
                log_str += 'policy mix rate = %.2f' % self.policy_mix_rate
                log_str += ', loss = %.4f' % avg_loss
                log_str += ', success rate = %.1f' % avg_success_rate
                log_str += ', distance (get tasks only) = %.2f' % avg_distance
                log_str += ', num interactions = %d / %d' % \
                    (total_interactions, total_steps)

                logging.info('')
                logging.info(log_str)

                # Save last student's model
                student.save('last')

                # Save best student's model
                eval_success_rate, eval_info = \
                    self.evaluate(datasets['dev'], world, student, teacher)
                if eval_success_rate > best_eval_success_rate:
                    logging.info('New best success rate: %.1f' %
                        eval_success_rate)
                    best_eval_success_rate = eval_success_rate
                    student.save('best_dev')
                    traj_file_path = os.path.join(
                        self.config.experiment_dir, 'best_dev.traj')
                    self.save_eval_info(traj_file_path, eval_info)

            if decay_every is not None and i_iter % decay_every == 0:
                self.policy_mix_rate = 0.9**(i_iter // decay_every)
                logging.info('Decay policy mix rate to %.2f' %
                    self.policy_mix_rate)

    def evaluate(self, dataset, world, student, teacher, save_traj=False):

        student.prepare(world)

        eval_info = {}
        total_success = (0, 0)
        total_distance = (0, 0)

        for i, batch in enumerate(dataset.iterate_batches()):

            with torch.no_grad():
                info = self.do_rollout(batch, world, student, teacher, True)
                success = info['success']
                distances = info['distances']
                action_seqs = info['action_seqs']

            total_success = util.add_stat(total_success, success)
            total_distance = util.add_stat(total_distance, distances)

            assert len(batch) == len(action_seqs)
            zipped_info = zip(batch, action_seqs, success)
            for item, traj, is_success in zipped_info:
                assert item['id'] not in eval_info
                eval_info[item['id']] = {
                        'actions': traj,
                        'success': int(is_success),
                    }

        for instance in dataset:
            assert instance['id'] in eval_info, instance['id']

        success_rate = total_success[0] / total_success[1] * 100
        avg_distance = total_distance[0] / total_distance[1]

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += 'success rate = %.1f' % success_rate
        log_str += ', distance (get tasks only) = %.2f' % avg_distance

        logging.info(log_str)

        if save_traj:
            traj_file_path = os.path.join(
                self.config.experiment_dir, dataset.split + '.traj')
            self.save_eval_info(traj_file_path, eval_info)

        return success_rate, eval_info

    def save_eval_info(self, file_path, eval_info):
        with open(file_path, 'w') as f:
            json.dump(eval_info, f)
        logging.info('Saved eval info to %s' % file_path)

