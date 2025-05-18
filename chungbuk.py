# reinforcement_learning.py
# -*- coding: utf-8 -*-
"""
Enhanced Reinforcement Learning Script with Q/V Table Display
Sections:
1. Environment & Agent\ n2. Q-value computation & display
3. V-value computation & display
4. Policy extraction & display
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
        if self.labels[x][y] == 'goal':
            return agent.pos.copy(), self.GOAL, True
        if not (0 <= nx < self.rewards.shape[0] and 0 <= ny < self.rewards.shape[1]):
            return agent.pos.copy(), self.CLIFF, True
        agent.pos = np.array([nx, ny])
        reward = self.rewards[nx, ny]
        done = (self.labels[nx][ny] == 'goal')
        return agent.pos.copy(), reward, done

class Agent:
    actions = np.array([[-1,0], [0,1], [1,0], [0,-1]])  # up, right, down, left
    prob = np.full(4, 0.25)

    def __init__(self, start_pos):
        self.pos = np.array(start_pos)

    def reset(self, pos):
        self.pos = np.array(pos)

# --- Q-value computation ---
def compute_q(env, agent, depth_limit, gamma=0.9):
    Q = np.zeros((*env.rewards.shape, len(agent.actions)))
    def q_val(pos, action, depth):
        agent.reset(pos)
        if env.labels[pos[0]][pos[1]] == 'goal':
            return env.GOAL
        _, r, done = env.move(agent, action)
        if depth == depth_limit or done:
            return r
        future = sum(agent.prob[a] * q_val(agent.pos.copy(), a, depth+1)
                     for a in range(len(agent.actions)))
        return r + gamma * future

    for i in range(env.rewards.shape[0]):
        for j in range(env.rewards.shape[1]):
            for a in range(len(agent.actions)):
                Q[i,j,a] = q_val((i,j), a, 0)
    return Q

# --- V-value computation ---
def compute_v(env, agent, depth_limit, gamma=0.85):
    V = np.zeros(env.rewards.shape)
    def v_val(pos, depth):
        agent.reset(pos)
        if env.labels[pos[0]][pos[1]] == 'goal':
            return env.GOAL
        if depth == depth_limit:
            return sum(agent.prob[a] * env.rewards[tuple(agent.pos)]
                       for a in range(len(agent.actions)))
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

# --- Benchmark V runtimes ---
def benchmark_v(env, agent, depths):
    times = []
    for d in depths:
        start = time.time()
        compute_v(env, agent, d)
        times.append(time.time() - start)
    return times

# --- Display utilities ---
def show_q_table(Q):
    rows, cols, _ = Q.shape
    border = '+' + '-------+' * cols
    for i in range(rows):
        print(border)
        for a in range(4):  # action rows
            line = '|'
            for j in range(cols):
                val = Q[i,j,a]
                line += f" {val:6.2f} |"
            print(line)
    print(border)


def show_v_table(V):
    rows, cols = V.shape
    border = '+' + '-------+' * cols
    print(border)
    for i in range(rows):
        line = '|'
        for j in range(cols):
            line += f" {V[i,j]:6.2f} |"
        print(line)
        print(border)

# --- Main ---
def main():
    env = Environment()
    agent = Agent((0,0))
    depths = list(range(8))

    # Q-tables and policies
    for d in depths:
        print(f"--- depth = {d} ---")
        Q = compute_q(env, agent, d)
        print("Q-table:")
        show_q_table(Q)
        print("Policy (arrows):")
        for row in extract_policy(Q):
            print(' '.join('↑→↓←'[a] for a in row))
        print()

    # V-value benchmarking and display
    times = benchmark_v(env, agent, depths)
    for d, t in zip(depths, times):
        print(f"depth={d}, time={t:.2f}s")
    V = compute_v(env, agent, depths[-1])
    print("V-table (depth={depths[-1]}):")
    show_v_table(V)

    # Reward table
    print("Reward map:")
    show_v_table(env.rewards)

    # Plot benchmark
    plt.plot(depths, times, marker='o')
    plt.xlabel('depth limit')
    plt.ylabel('time (s)')
    plt.title('V computation runtime')
    plt.show()

if __name__ == '__main__':
    main()
