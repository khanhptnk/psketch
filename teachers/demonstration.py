import os
import sys


class DemonstrationTeacher(object):

    def __init__(self, config):
        self.config = config

    def __call__(self, task, state):

        #print(state.render())
        #print(task)

        subtask = self.find_incomplete_subtask(task, state)

        actions = state.world.actions

        if subtask is None:
            return actions.STOP.index

        assert subtask.goal_name in ['use', 'go']

        if subtask.goal_name == 'use':
            return actions.USE.index

        _, best_action_seq = self.find_closest_resources(subtask, state)

        if best_action_seq is None:
            return actions.STOP.index

        ref_action = best_action_seq[0]

        return ref_action

    def find_incomplete_subtask(self, task, state):
        task_done = state.satisfies(task)
        if task_done:
            return None
        if task.subtasks is None:
            return task

        for subtask in task.subtasks[:-1]:
            incomplete_subtask = self.find_incomplete_subtask(subtask, state)
            if incomplete_subtask is not None:
                return incomplete_subtask

        incomplete_subtask = self.find_incomplete_subtask(task.subtasks[-1], state)
        assert incomplete_subtask is not None
        return incomplete_subtask

    def find_closest_resources(self, task, state):
        best_goal = (None, None)
        for goal_pos in state.find_resource_positions(task.goal_arg):
            action_seq = self.shortest_path(state, goal_pos)
            if best_goal[1] is None or len(action_seq) < len(best_goal[1]):
                best_goal = (goal_pos, action_seq)

        return best_goal


    def shortest_path(self, state, goal_pos):

        world = state.world
        nav_grid = state.make_navigation_grid()

        prev = {}
        queue = [None] * 1000
        start = 0
        end = 0
        new_item = (state.pos, state.dir)
        queue[end] = new_item
        end += 1
        prev[new_item] = -1

        while start < end:
            item = queue[start]
            start += 1

            pos, dir = item

            # Stop if facing the goal
            coord_change = world.action_space[dir].coord_change
            new_pos = (pos[0] + coord_change[0], pos[1] + coord_change[1])
            if new_pos == goal_pos:
                action_seq = []
                while prev[item] != -1:
                    pos, dir = item
                    action, item = prev[item]
                    action_seq.append(action)
                action_seq = list(reversed(action_seq))
                return action_seq

            for i, action in enumerate(world.action_space):

                if action in [world.actions.USE, world.actions.STOP]:
                    continue

                coord_change = action.coord_change
                new_pos = (pos[0] + coord_change[0], pos[1] + coord_change[1])
                new_dir = i

                # only change direction if new position is not enterable
                if nav_grid[new_pos[0], new_pos[1]]:
                    new_pos = pos

                new_item = (new_pos, new_dir)
                if new_item not in prev:
                    queue[end] = new_item
                    end += 1
                    prev[new_item] = (action.index, item)

        return None

