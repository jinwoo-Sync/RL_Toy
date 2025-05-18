# reinforcement_learning.py
# -*- coding: utf-8 -*-
"""
Modular Reinforcement Learning Script

Sections:
1. Environment & Agent
2. Depth-limited Action-value (Q) computation
3. Depth-limited State-value (V) computation
4. Policy extraction
5. Benchmarking V runtimes
6. Visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import time

class Environment:
    """Defines gridworld dynamics and rewards."""
    CLIFF, ROAD, SINK, GOAL = -3, -1, -2, 2

    def __init__(self):
        self.rewards = np.array([
            [self.ROAD, self.ROAD, self.ROAD, self.ROAD],
            [self.ROAD, self.ROAD, self.SINK, self.ROAD],
            [self.ROAD, self.ROAD, self.ROAD, self.GOAL]
        ])
        self.labels = [
            ['road','road','road','road'],
            ['road','road','sink','road'],
            ['road','road','road','goal']
        ]

    def move(self, agent, action_idx):
        x, y = agent.pos
        dx, dy = agent.actions[action_idx]
        nx, ny = x + dx, y + dy
        # terminal check
        if self.labels[x][y] == 'goal':
            return agent.pos.copy(), self.GOAL, True
        # out-of-bounds
        if nx<0 or nx>=self.rewards.shape[0] or ny<0 or ny>=self.rewards.shape[1]:
            return agent.pos.copy(), self.CLIFF, True
        # valid
        agent.pos = np.array([nx,ny])
        reward = self.rewards[nx,ny]
        done = (self.labels[nx][ny] == 'goal')
        return agent.pos.copy(), reward, done

class Agent:
    """Holds position and available actions."""
    actions = np.array([[-1,0],[0,1],[1,0],[0,-1]])  # up,right,down,left
    prob = np.full(4, 0.25)

    def __init__(self, start_pos):
        self.pos = np.array(start_pos)

    def reset(self, pos):
        self.pos = np.array(pos)

# --- Action-value (Q) ---
def compute_q(env, agent, depth_limit, gamma=0.9):
    Q = np.zeros((*env.rewards.shape, len(agent.actions)))

    def q_val(pos, action, depth):
        agent.reset(pos)
        if env.labels[pos[0]][pos[1]] == 'goal':
            return env.GOAL
        _, r, done = env.move(agent, action)
        if depth == depth_limit or done:
            return r
        future = 0.0
        next_pos = agent.pos.copy()
        for a in range(len(agent.actions)):
            future += agent.prob[a] * q_val(next_pos, a, depth+1)
        return r + gamma * future

    for i in range(env.rewards.shape[0]):
        for j in range(env.rewards.shape[1]):
            for a in range(len(agent.actions)):
                Q[i,j,a] = q_val((i,j), a, 0)
    return Q

# --- State-value (V) ---
def compute_v(env, agent, depth_limit, gamma=0.85):
    V = np.zeros(env.rewards.shape)

    def v_val(pos, depth):
        agent.reset(pos)
        if env.labels[pos[0]][pos[1]] == 'goal':
            return env.GOAL
        if depth == depth_limit:
            val = 0.0
            for a in range(len(agent.actions)):
                _, r, _ = env.move(agent, a)
                val += agent.prob[a] * r
            return val
        val = 0.0
        for a in range(len(agent.actions)):
            _, r, done = env.move(agent, a)
            next_pos = agent.pos.copy()
            if done:
                val += agent.prob[a] * r
            else:
                val += agent.prob[a] * (r + gamma * v_val(next_pos, depth+1))
        return val

    for i in range(env.rewards.shape[0]):
        for j in range(env.rewards.shape[1]):
            V[i,j] = v_val((i,j), 0)
    return V

# --- Policy extraction ---
def extract_policy(Q):
    return np.argmax(Q, axis=-1)

# --- Benchmarking V ---
def benchmark_v(env, agent, depths):
    times = []
    for d in depths:
        start = time.time()
        compute_v(env, agent, d)
        times.append(time.time() - start)
    return times

# --- Visualization ---
def show_policy(policy):
    arrows = ['↑','→','↓','←']
    for row in policy:
        print(' '.join(arrows[a] for a in row))

# --- Main ---
def main():
    env = Environment()
    agent = Agent((0,0))
    depths = list(range(8))

    for d in depths:
        Q = compute_q(env, agent, d)
        print(f"Policy Q depth={d}")
        show_policy(extract_policy(Q))

    times = benchmark_v(env, agent, depths)
    plt.plot(depths, times, marker='o')
    plt.xlabel('depth')
    plt.ylabel('time (s)')
    plt.show()

if __name__ == '__main__':
    main()
