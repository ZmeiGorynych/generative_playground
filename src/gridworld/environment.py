import random
import numpy as np


def random_position(max_height, max_width):
    return np.random.randint(0, max_height), np.random.randint(0, max_width)


def compare(x, y):
    return np.array_equal(x, y)


class Gridworld:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.initial_state = None
        self.initial_player_pos = None
        self.state = np.zeros((height, width, 4), dtype=int)
        self.player = np.array([0, 0, 0, 1])
        self.wall = np.array([0, 0, 1, 0])
        self.pit = np.array([0, 1, 0, 0])
        self.goal = np.array([1, 0, 0, 0])
        self.player_position = None

        self.up = 0
        self.right = 1
        self.down = 2
        self.left = 3

    def set_player_position(self, position):
        if self.player_position is not None:
            self.state[tuple(self.player_position)] = np.zeros(4)
        self.player_position = tuple(position)
        self.state[tuple(position)] = self.player

    def save_state(self):
        self.initial_state = np.copy(self.state)
        self.initial_player_pos = np.copy(self.player_position)

    @classmethod
    def deterministic_easy(cls, height=4, width=4):
        world = cls(height, width)
        # place player
        world.set_player_position((0, 1))
        # place goal
        world.state[3, 3] = world.goal
        world.save_state()
        return world

    @classmethod
    def deterministic(cls, height=4, width=4):
        world = cls(height, width)
        # place player
        world.set_player_position((0, 1))
        # place wall
        world.state[2, 2] = world.wall
        # place pit
        world.state[1, 1] = world.pit
        # place goal
        world.state[3, 3] = world.goal
        world.save_state()
        return world

    @classmethod
    def random_player_pos(cls, height=4, width=4):
        world = cls(height, width)
        pos = random_position(height, width)
        while pos in ((2, 2), (1, 1), (1, 2)):
            pos = random_position(height, width)
        # place player
        world.set_player_position(pos)
        # place wall
        world.state[2, 2] = world.wall
        # place pit
        world.state[1, 1] = world.pit
        # place goal
        world.state[1, 2] = world.goal
        world.save_state()
        return world

    @classmethod
    def random(cls, height=4, width=4):
        world = cls(height, width)
        places = [(i, j) for i in range(height) for j in range(width)]
        positions = random.sample(places, 4)
        # place player
        world.set_player_position(positions[0])
        # place wall
        world.state[positions[1]] = world.wall
        # place pit
        world.state[positions[2]] = world.pit
        # place goal
        world.state[positions[3]] = world.goal
        world.save_state()
        return world

    def step(self, action):
        diff = (0, 0)
        if action == self.up and self.player_position[0] > 0:
            diff = (-1, 0)
        elif action == self.right and self.player_position[1] < self.width - 1:
            diff = (0, 1)
        elif action == self.down and self.player_position[0] < self.height - 1:
            diff = (1, 0)
        elif action == self.left and self.player_position[1] > 0:
            diff = (0, -1)

        old_pos = tuple(np.copy(self.player_position))
        new_pos = tuple(np.add(self.player_position, diff))
        done = False
        reward = -1
        if compare(self.state[new_pos], self.wall):
            new_pos = old_pos
        elif compare(self.state[new_pos], self.pit):
            done = True
            reward = -10
        elif compare(self.state[new_pos], self.goal):
            done = True
            reward = 10
        old_state = np.copy(self.state)
        self.set_player_position(new_pos)
        return old_state, reward, self.state, done

    def reset(self):
        self.state = np.copy(self.initial_state)
        self.player_position = np.copy(self.initial_player_pos)
        return self.state

    def display(self):
        grid = np.empty((self.height, self.width), dtype=str)
        for i in range(self.height):
            for j in range(self.width):
                point = self.state[i, j]
                if compare(point, self.player):
                    grid[i, j] = '@'
                elif compare(point, self.wall):
                    grid[i, j] = 'W'
                elif compare(point, self.goal):
                    grid[i, j] = '+'
                elif compare(point, self.pit):
                    grid[i, j] = '^'
                else:
                    grid[i, j] = ' '
        return grid
