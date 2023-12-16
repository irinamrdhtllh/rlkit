import random
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


@dataclass
class State:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))


class Env(ABC):
    @abstractproperty
    def action_space(self):
        pass

    @abstractproperty
    def state_space(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass


class SimpleGrid(Env):
    class Action(Enum):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

    def __init__(
        self,
        row_size: int,
        col_size: int,
        n_walls: int,
        start: tuple[int, int],
        terminal: tuple[int, int],
    ):
        self.row_size = row_size
        self.col_size = col_size
        self.n_walls = n_walls
        self.start = start
        self.terminal = terminal

        self.walls = []
        for _ in range(n_walls):
            loc = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
            while loc == start or loc == terminal:
                loc = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
            self.walls.append(loc)

        self.rewards = {terminal: +1}
        hole = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
        while hole == start or hole == terminal or hole in self.walls:
            hole = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
        self.rewards[hole] = -1

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def state_space(self):
        return VectorState(2)

    def step(self, action):
        x = self.state_space.data[0]
        y = self.state_space.data[1]
        if action == SimpleGrid.Action.UP:
            if y != 0:
                y -= 1
        elif action == SimpleGrid.Action.RIGHT:
            if y != self.col_size - 1:
                x += 1
        elif action == SimpleGrid.Action.DOWN:
            if y != self.row_size - 1:
                y += 1
        elif action == SimpleGrid.Action.LEFT:
            if x != 0:
                x -= 1

        if (x, y) not in self.walls:
            self.state_space.data[0] = x
            self.state_space.data[1] = y

        state = (self.state_space.data[0], self.state_space.data[1])
        try:
            reward = self.rewards[state]
        except KeyError:
            reward = 0

        done = False
        if state == self.terminal:
            done = True

        return self.state_space, reward, done

    def reset(self):
        self.state_space.data = self.start
        return self.state_space


class ActionSpace(ABC):
    @abstractmethod
    def random(self):
        pass


class Discrete(ActionSpace):
    def __init__(self, size: int):
        self.size = size

    def random(self):
        return random.randint(0, self.size - 1)


class StateSpace(ABC):
    pass


class VectorState(StateSpace):
    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(self.size)


MAX_STEP = 100


def inference():
    env = SimpleGrid(4, 4, 4, (3, 0), (0, 3))
    done = False
    total_reward = 0
    step = 0
    state = env.reset()
    while not done and step < MAX_STEP:
        action = env.action_space.random()
        state, reward, done = env.step(action)
        total_reward += reward
        step += 1
    if done:
        print("Finish")
    else:
        print("Timeout")
    print(f"Total reward: {total_reward}")
    print(f"Total step: {step}")


if __name__ == "__main__":
    inference()
    # print(get_grid_walls())

    # action_space = Discrete(5)
    # action = action_space.random()
    # # print(action)

    # state_space = VectorState(2)
    # state = state_space.data
    # print(state)
