import sys
import os
file_path = os.path.dirname(__file__)
pplib_path = os.path.join(file_path, "..", "build")
sys.path.append(pplib_path)
from solver import SolverWrapper
from copy import copy

import numpy as np

# 读写随机提供的地图文件，并转换成输入
def read_map(fname):
    world_len =  world_wid = 0
    
    world = -1
    with open(fname, 'r') as fp:
        for i in range(4):
            l = fp.readline()
            if i == 1:
                words = l.strip().split()
                world_len = int(words[-1])
            if i== 2:
                words = l.strip().split()
                world_wid = int(words[-1])
        
        world = np.zeros([world_len, world_wid])
        row = 0
        while True:
            line = fp.readline()
            if not line:
                break
            for j in range(world_wid):
                if line[j] != '.':
                    world[row, j] = -1
            row += 1

    return world_len, world_wid, world

# 读写随机提供的场景文件，并转换成输入
def read_scen(fname, num_agents = 50):
    starts = np.zeros([num_agents, 2], dtype=int)
    goals = np.zeros([num_agents, 2], dtype=int)
    with open(fname, 'r')  as fp:
        fp.readline()
        for i in range(num_agents):
            line = fp.readline()
            words = line.split()
            starts[i, 0] = int(words[5])
            starts[i, 1] = int(words[4])
            goals[i, 0] = int(words[7])
            goals[i, 1] = int(words[6])
    return starts, goals

def normalize_input(world : np.ndarray, starts : np.ndarray, goals : np.ndarray):
    world_len, world_width = world.shape
    num_agents = starts.shape[0]
    goals_ = np.zeros([world_len, world_wid])
    for i in range(num_agents):
        world[int(starts[i, 0]), int(starts[i, 1])] = i + 1
        goals_[int(goals[i, 0]), int(goals[i, 1])] = i + 1
    return world, goals_

def translate_order2edges(ordered_agents):
    num_agents = len(ordered_agents)
    priorities = np.zeros( [num_agents - 1, 2])
    for i in range(num_agents - 1) :
        priorities[i, 0] = ordered_agents[i]
        priorities[i, 1] = ordered_agents[i + 1]
    return priorities

if __name__ == '__main__':
    map_fname = os.path.join(file_path, "..", "random-32-32-20.map")
    world_len, world_wid, world = read_map(map_fname)
    scen_fname = os.path.join(
        file_path, "..", "random-32-32-20-random-1.scen")
    starts, goals = read_scen(scen_fname)
    num_agents = starts.shape[0]
    world, goals = normalize_input(world, starts, goals)
    screen = 3
    time_limit = 60.0
    solver = SolverWrapper(world, world_len, world_wid, goals, 
                           num_agents, screen, time_limit)
    solver.step(np.array([2.0,1.0]), 2)
    solutions = solver.getSolutions()
    print('solutions:' , solutions)
    # validate
    solutions_ = copy(solutions)
    solutions_ = solutions_.reshape([num_agents, -1, 2])
    solutions_[:, :, 0] = solutions_[:, :, 0] * world_wid + solutions_[:, :, 1]
    solutions_ = solutions_[:, :, 0].reshape([num_agents, -1])
    solutions_[solutions_<0] = -1
    succ = solver.validateSolutions(solutions_, -1)
    print("succ? ", succ)


