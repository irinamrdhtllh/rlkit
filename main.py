import os
import random
import time
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Env(ABC):
    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def state_space(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def visualize(self):
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
        max_step: int = 100,
    ):
        self.row_size = row_size
        self.col_size = col_size
        self.n_walls = n_walls
        self.start = start
        self.terminal = terminal
        self._action_space = Discrete(4)
        self._state_space = VectorState(2)
        self.terminated = False
        self.truncated = False
        self.timestep = 0
        self.max_step = max_step

        self.walls = []
        for _ in range(n_walls):
            loc = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
            while loc == start or loc == terminal:
                loc = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
            self.walls.append(loc)

        self.rewards = {terminal: +1}
        self.hole = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
        while self.hole == start or self.hole == terminal or self.hole in self.walls:
            self.hole = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
        self.rewards[self.hole] = -1

    @property
    def action_space(self):
        return self._action_space

    @property
    def state_space(self):
        return self._state_space

    def step(self, action):
        if self.terminated:
            raise Exception("Env has reached the terminal state")
        if self.truncated:
            raise Exception("Env has reached the maximum timestep")

        self.timestep += 1
        action = SimpleGrid.Action(action)
        i, j = self._state_space.data
        if action == SimpleGrid.Action.UP:
            if i != 0:
                i -= 1
        elif action == SimpleGrid.Action.RIGHT:
            if j != self.col_size - 1:
                j += 1
        elif action == SimpleGrid.Action.DOWN:
            if i != self.row_size - 1:
                i += 1
        elif action == SimpleGrid.Action.LEFT:
            if j != 0:
                j -= 1

        if (i, j) not in self.walls:
            self._state_space.data = np.array((i, j))

        state = tuple(self._state_space.data)
        try:
            reward = self.rewards[state]
        except KeyError:
            reward = 0

        if state == self.terminal:
            self.terminated = True
        elif self.timestep >= self.max_step:
            self.truncated = True

        return self._state_space, reward, self.terminated, self.truncated

    def reset(self):
        self._state_space.data = self.start
        return self._state_space

    def visualize(self):
        for i in range(self.row_size):
            for j in range(self.col_size):
                if (i, j) in self.walls:
                    print("#", end="")
                elif (i, j) == tuple(self.state_space.data):
                    print("x", end="")
                elif (i, j) == self.hole:
                    print("o", end="")
                else:
                    print(".", end="")
            print()
        time.sleep(0.25)
        if not self.terminated or not self.truncated:
            os.system("cls")


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
    env = SimpleGrid(4, 4, 4, (3, 0), (0, 3), 100)
    done = False
    total_reward = 0
    step = 0
    state = env.reset()
    while not done:
        action = env.action_space.random()
        state, reward, terminated, truncated = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        env.visualize()
    if terminated:
        print("Finish")
    if truncated:
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
