import random
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class State:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))


GRID = """\
..#.
..#.
#...
...#
"""


def get_grid_walls() -> list[State]:
    walls: list[State] = []
    for y, line in enumerate(GRID.splitlines()):
        for x, char in enumerate(line):
            if char == "#":
                walls.append(State(x, y))
    return walls


GRID_HEIGHT = len(GRID.splitlines())
GRID_WIDTH = len(GRID.splitlines()[0])
MAX_STEP = 100

start = State(x=3, y=0)
finish = State(x=0, y=3)

rewards = {
    State(0, 3): +2,
    State(1, 1): -1,
}


class Action(Enum):
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()


def get_action(state: State) -> Action:
    return random.choice(list(Action))


def inference():
    state = start
    total_reward = 0
    step = 0
    while state != finish and step < MAX_STEP:
        action = get_action(state)
        if action == Action.UP:
            if state.y != 0:
                state.y -= 1
        elif action == Action.RIGHT:
            if state.y != GRID_WIDTH - 1:
                state.x += 1
        elif action == Action.DOWN:
            if state.y != GRID_HEIGHT - 1:
                state.y += 1
        elif action == Action.LEFT:
            if state.x != 0:
                state.x -= 1
        try:
            reward = rewards[state]
        except KeyError:
            reward = 0
        total_reward += reward
        step += 1
    if state == finish:
        print("Finish")
    else:
        print("Timeout")
    print(f"Total reward: {total_reward}")
    print(f"Total step: {step}")


if __name__ == "__main__":
    inference()
    # print(get_grid_walls())
