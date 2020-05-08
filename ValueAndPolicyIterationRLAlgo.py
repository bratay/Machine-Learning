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

# # each state has 4 possible next states 4 directions:
# position = [0,1]
# grid = [[0, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, 0]]

# # get Vπ(s) in two parts
# Vs = 0

# # First
# # multiplying each possible reward by its probability and adding them together
# # Rt·p(rup|st,at) + Rt·p(rdown|st,at) + Rt·p(rright|st,at) + Rt·p(rleft|st,at)
# # for all actions p(rt+1|st,at) = 0.25
# # the first: -1·0.25 + -1·0.25 + -1·0.25 + -1·0.25 = -1
# avgReward = -1

# # Second
# # multiplying each of the neighboring Vπ(s’) by its probability and adding them together
# # p(st+1|st,at) = 0.25 for all actions and that γ = 1.
# # SO: Vπ(s’up)·0.25 + Vπ(s’down)·0.25 + Vπ(s’right)·0.25+ Vπ(s’left)·0.25
# discountedAvg = 0

# Vs = avgReward + discountedAvg

# # repeat 1 and 2
# while():
#     # multiplying each possible reward by its probability and adding them together
#     # Rt·p(rup|st,at) + Rt·p(rdown|st,at) + Rt·p(rright|st,at) + Rt·p(rleft|st,at)
#     avgReward = -1

#     # multiplying each of the neighboring Vπ(s’) by its probability and adding them together
#     # Vπ(s’up)·0.25 + Vπ(s’down)·0.25 + Vπ(s’right)·0.25+ Vπ(s’left)·0.25
#     discountedAvg = 0

#     Vs = avgReward + discountedAvg

# ########################################
# # Value Iteration RL Algorithm
# ########################################

gamma = 0.5
alpha = 0.5
terminationStates = [[0, 0], [4, 4]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 10000


vFunction = np.zeros((5, 5))
deltas = {(i, j): list() for i in range(5) for j in range(5)}
states = [[i, j] for i in range(5) for j in range(5)]


def generateInitialState():
    initial = random.choice(states[1:-1])
    return initial


def generateNextAction():
    return random.choice(actions)


def takeAction(state, action):
    if list(state) in terminationStates:
        return 0, None
    final = np.array(state)+np.array(action)

    # Out of bounds
    if -1 in list(final) or 5 in list(final):
        final = state
    return -1, list(final)


for it in range(0, numIterations):
    state = generateInitialState()
    while True:
        action = generateNextAction()
        reward, final = takeAction(state, action)

        if final is None:
            break

        before = vFunction[state[0], state[1]]
        vFunction[state[0], state[1]] += .5 * \
            (reward + .5*vFunction[final[0], final[1]] -
             vFunction[state[0], state[1]])
        deltas[state[0], state[1]].append(
            float(np.abs(before-vFunction[state[0], state[1]])))

        state = final


print(vFunction)
