import sys
sys.path.append('..')
import itertools

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

        #ID = 0
        #print(states[ID].render())

        timer = self.config.trainer.max_timesteps
        done = [False] * batch_size
        success = [False] * batch_size
        trajectories = [[] for i in range(batch_size)]

        while not all(done) and timer > 0:
            actions = student.act(states)

            ref_actions = [None] * batch_size

            for i in range(batch_size):
                # Ask teacher for reference action
                if not is_eval:
                    if done[i]:
                        ref_actions[i] = -1
                    else:
                        ref_actions[i] = teacher(tasks[i], states[i])

                #actions[i] = ref_actions[i]

                terminate = done[i] or actions[i] == world.actions.STOP.index

                # Transition to next state
                if terminate:
                    success[i] = states[i].satisfies(tasks[i])
                    assert success[i] is not None
                else:
                    _, states[i] = states[i].step(actions[i])

                done[i] |= terminate

                # Save trajectory
                trajectories[i].append(actions[i])

            # Receive reference actions
            if not is_eval:
                student.receive(ref_actions)

            timer -= 1

        distances = []
        for i in range(batch_size):
            if not done[i]:
                success[i] = states[i].satisfies(tasks[i])
            if not success[i] and tasks[i].goal_name == 'get':
                state = world.init_state(
                    batch[i]['grid'], states[i].pos, states[i].dir)
                _, best_action_seq = teacher.find_closest_resources(
                    tasks[i], state)
                distances.append(len(best_action_seq))

        #print(sum(success) / len(success))
        #print(states[ID].render())
        #print(trajectories[ID])
        #print(tasks[ID])

        return trajectories, success, distances

    def train(self, datasets, world, student, teacher):

        student.prepare(world)

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0
        total_success = (0, 0)
        total_distance = (0, 0)
        best_eval_success_rate = 1e9

        for batch in itertools.cycle(datasets['train'].iterate_batches()):

            i_iter += 1

            trajectories, success, distances = \
                self.do_rollout(batch, world, student, teacher, False)

            total_success = util.add_stat(total_success, success)
            total_distance = util.add_stat(total_distance, distances)

            loss = student.learn()
            total_loss += loss

            if i_iter % log_every == 0:

                avg_loss = total_loss / log_every
                avg_success_rate = total_success[0] / total_success[1] * 100
                avg_distance = total_distance[0] / total_distance[1]

                total_loss = 0

                training_status_str = '%s (%d %d%%)' % (
                    util.time_since(self.config.start_time, i_iter / max_iters),
                    i_iter, i_iter / max_iters * 100)

                log_str = '\n%s Train iter %d: ' % (training_status_str, i_iter)
                log_str += 'loss = %.4f' % avg_loss
                log_str += ', success rate = %.1f' % avg_success_rate
                log_str += ', distance = %.2f' % avg_distance

                print(log_str)

                # Save last student's model
                student.save('last')

                # Save best student's model
                eval_success_rate, eval_trajectories = \
                    self.evaluate(datasets['dev'], world, student, teacher)
                if eval_success_rate > best_eval_success_rate:
                    print('New best_dev: %.1f', eval_success_rate)
                    best_eval_success_rate = eval_success_rate
                    student.save('best_dev', trajectories=eval_trajectories)

    def evaluate(self, dataset, world, student, teacher):

        all_trajectories = {}
        total_success = (0, 0)
        total_distance = (0, 0)

        for i, batch in enumerate(dataset.iterate_batches()):
            with torch.no_grad():
                trajectories, success, distances = \
                    self.do_rollout(batch, world, student, teacher, True)

            total_success = util.add_stat(total_success, success)
            total_distance = util.add_stat(total_distance, distances)

            assert len(batch) == len(trajectories)
            for item, traj in zip(batch, trajectories):
                assert item['id'] not in all_trajectories
                all_trajectories[item['id']] = traj

        success_rate = total_success[0] / total_success[1] * 100
        avg_distance = total_distance[0] / total_distance[1]
        print('== Evaluation on %s ==' % dataset.split)
        print('Success rate = %.1f, distance = %.2f' %
            (success_rate, avg_distance))

        return success_rate, all_trajectories
