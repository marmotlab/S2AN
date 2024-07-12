from pysipps_planner import SippsPlanner

import sys
import os
file_path = os.path.dirname(__file__)

sys.path.append(os.path.join(file_path, ".."))
from map_generator import MapLoader, dump_solutions, generate_timestamp_prefix

import numpy as np

# 测试内容。使用lns算法看看。
# 简单的逆序算法


if __name__ == '__main__':
    map_fname = os.path.join(file_path, "..",
                             "PriorityPlanner", "random-32-32-20.map")
    scen_fname = os.path.join(
        file_path, "..", "PriorityPlanner", "random-32-32-20-random-1.scen")

    loader = MapLoader()
    num_agents = 150 # max 409
    world, goals = loader.load_movingai(scen_fname, map_fname, num_agents)

    planner = SippsPlanner(world, goals, num_agents, neighbor_size=3)
    num_iter = 0
    max_neighbors = 8
    while True:
        ei = planner.get_collision_edge_index()
        ei2 = planner.get_target_edge_index()
        if (ei.shape[0] == 0):
            break
        agent_set = set(ei[:, 0])
        replan_agents = []
        priorities = planner.get_priorities()

        for i in priorities:
            if i in agent_set and not( i in replan_agents) :
                replan_agents.append(i)
        replan_agents.reverse()

        print("replan agents: ", replan_agents[:max_neighbors])
        planner.step(replan_agents[:max_neighbors])
        solutions = planner.get_solutions()
        num_iter += 1

    # save the solutions
    fname = generate_timestamp_prefix()
    fpath = file_path+"/../results"
    dump_solutions(solutions, fpath, fname)
    loader.dump_movingai(world, goals, fpath, fname, num_agents)

    print("number of iterations of greedy reverse.", num_iter)
    




    
    

