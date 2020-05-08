# Load libraries
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import array
from numpy import cov
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import random

# ########################################
# # Policy Iteration RL Algorithm
# ########################################

vFunction = np.zeros((5, 5))
deltas = {(i, j): list() for i in range(5) for j in range(5)}
states = [[i, j] for i in range(5) for j in range(5)]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]


def takeAction(state, action):
    if list(state) == [0, 0] or list(state) == [4, 4]:
        return 0, None
    f = np.array(state)+np.array(action)

    # Out of bounds
    if -1 in list(f) or 5 in list(f):
        f = state
    return -1, list(f)


for i in range(0, 10000):
    state = random.choice(states[1:-1])

    while True:
        action = random.choice(actions)
        reward, f = takeAction(state, action)

        if f is None:
            break

        pre = vFunction[state[0], state[1]]
        vFunction[state[0], state[1]] += .5 * \
            (reward + .5*vFunction[f[0], f[1]] -
             vFunction[state[0], state[1]])

        deltas[state[0], state[1]].append(
            float(np.abs(pre-vFunction[state[0], state[1]])))

        state = f

print(vFunction)


# ########################################
# # Value Iteration RL Algorithm
# ########################################
