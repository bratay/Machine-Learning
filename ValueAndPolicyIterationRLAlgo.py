from numpy import array
import numpy as np
import random

# ########################################
# # Policy Iteration RL Algorithm
# ########################################
print('\n\nPolicy Iteration RL optimal policy (π)')

vFunction = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        vFunction[i][j] = -1
vFunction[0,0] = 0
vFunction[4,4] = 0
deltas = {(i, j): list() for i in range(5) for j in range(5)}
states = [[i, j] for i in range(5) for j in range(5)]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]


def act(state, action):
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
        reward, f = act(state, action)

        if f is None:
            break

        pre = vFunction[state[0], state[1]]
        vFunction[state[0], state[1]] += .5 * \
            (reward + .5*vFunction[f[0], f[1]] -
             vFunction[state[0], state[1]])

        deltas[state[0], state[1]].append(
            float(np.abs(pre-vFunction[state[0], state[1]])))

        state = f

    if(i == 0):
        print('Iteration 0')
        print(vFunction)
    if(i == 1):
        print('Iteration 1')
        print(vFunction)
    if(i == 10):
        print('Iteration 10')
        print(vFunction)


print("Final Iteration")
print(vFunction)


# ########################################
# # Value Iteration RL Algorithm
# ########################################
print('\n\nValue Iteration RL optimal policy (π)')
print('--------------------------------')

vFunction = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        vFunction[i][j] = -1
vFunction[0,0] = 0
vFunction[4,4] = 0

deltas = {(i, j): list() for i in range(5) for j in range(5)}
states = [[i, j] for i in range(5) for j in range(5)]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]


def act(state, action):
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
        reward, f = act(state, action)

        if f is None:
            break

        pre = vFunction[state[0], state[1]]
        vFunction[state[0], state[1]] += .5 * \
            (reward + .5*vFunction[f[0], f[1]] -
             vFunction[state[0], state[1]])

        deltas[state[0], state[1]].append(
            float(np.abs(pre-vFunction[state[0], state[1]])))

        state = f

    if(i == 0):
        print('Iteration 0')
        print(vFunction)
    if(i == 1):
        print('Iteration 1')
        print(vFunction)
    if(i == 10):
        print('Iteration 10')
        print(vFunction)


print("Final Iteration")
print(vFunction)
