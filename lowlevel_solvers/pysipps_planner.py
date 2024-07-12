import sys
import os
file_path = os.path.dirname(__file__)
pplib_path = os.path.join(file_path, "..", "MAPF-LNS2", "build")
sys.path.append(pplib_path)

from solver import SolverWrapper as Solver

import numpy as np
# import yaml

# add a wrapper for SIPPS planner based on c++.
class SippsPlanner:
    def __init__(self,
                 world:np.ndarray, 
                 goals : np.ndarray, 
                 num_agents : int, 
                 max_num_iter : int = 100,
                 time_limit: float = 60 ,
                 neighbor_size: int = 8,
                 screen=3):
        """python wrapper for c++ sipps planner . legalize the input.

        Args:
            world (np.ndarray): [world_wid, world_height] as like scrimp. -1: obstacle. 0 free. >0 agent start location
            goals (np.ndarray): [world_wid, world_height] as like scrimp.
            num_agents (int): num_of_agents
            screen (int, optional):  screen print information level. Defaults to 3.
        """

        self.planner = Solver(
            world.astype(np.double), int(world.shape[0]), int(world.shape[1]), 
            goals.astype(np.double), int(num_agents), int(max_num_iter), 
            int(neighbor_size) , int(screen), float(time_limit))
        
        self.num_agents = num_agents

        self.world_wid = int(world.shape[0])
        self.world_height = int(world.shape[1])
        self.world = world
        self.goals = goals
    
    def step(self, replan_seq):
        if (not isinstance(replan_seq, np.ndarray)):
            replan_seq = np.array(replan_seq, dtype=np.double)
        replan_seq = replan_seq.astype(np.double)       
        num_seq = replan_seq.shape[0]
        self.planner.step(replan_seq, num_seq)

    def solveSingleAgent(self, id, higher_agnets):
        ''' solve one agent's trajectory by using SIPPS to avoid the higher agents.
        The higher agents should be solved before.
        '''
        return self.planner.solveSingleAgent(
            int(id), np.array(higher_agnets).astype(int).tolist())

    def repairSolutions(self):
        return self.planner.repairSolutions()

    def get_priorities(self) -> np.ndarray:
        """priority sequence. Equal to planned agents.
        """
        return self.planner.getPriorities()

    def get_target_edge_index(self)->np.ndarray:
        ''' get the target blocking matrix ecoded by the edge index format.
        '''
        return self.planner.getTragetEdgeIndex()
    
    def get_target_matrix(self):
        ''' get the target matrix as in paper. matrix[i, j] means i is blocked by j.
        '''
        target_ei = self.get_target_edge_index()
        target_matrix = np.zeros([self.num_agents, self.num_agents])
        for i in range(target_ei.shape[0]):
            target_matrix[target_ei[i,0], target_ei[i,1]] = 1
        return target_matrix
    
    def get_solutions(self) -> np.ndarray:
        """cannot use when solve return false. incomplete if not solved.

        Returns:
            np.ndarray: [num_agents, makespan, 2]. -1 represent when comes to the end or not solved yet.
        """
        res = self.planner.getSolutions().reshape([self.num_agents, -1 , 2])

        return res
    
    def get_collision_edge_index(self) -> np.ndarray:
        """cannot use when solve return false. incomplete if not solved.

        Returns:
            np.ndarray: [num_collision_edges, 2]. colliding agent index pairs.
        """
        res = self.planner.getCollisionEdgeIndex()

        return res

    def locate_end(self, solutions : np.ndarray):
        # solution[a, end_time] is valid. solution[a, end_time+1]=-1
        num_agents = solutions.shape[0]
        max_t = solutions.shape[1]
        end_times = np.zeros([num_agents]).astype(int)
        for i in range(solutions.shape[0]):
            index = np.where(solutions[i, :] < 0)[0]
            if (len(index) > 0):
                index = index[0] - 1
            else:
                index = max_t - 1
            end_times[i] = index
        return end_times
    
    def calcInitialSolution(self, init_seq=None):
        # calculate an initial solution by using a random priority.
        # unused in the paper.
        if init_seq is None:
            init_seq = np.arange(self.num_agents)
            np.random.shuffle(init_seq)
        assert len(init_seq) == self.num_agents, "init seq should be size of agents"
        
        init_seq = np.array(init_seq, dtype=np.double)
        self.planner.calcInitialSolution(init_seq, int(self.num_agents))

    def encode_solutions(self, solutions:np.ndarray, world_wid):
        solutions = solutions.copy()
        solutions[:, :, 0] = world_wid * solutions[:, :, 0] + solutions[:, :, 1] 
        solutions = solutions[:, :, 0]
        # solutions[solutions<0] = -1
        return solutions

    def validate_solutions(self,  solutions:np.ndarray, soc):
        solutions = self.encode_solutions(solutions, self.world_wid)
        solutions[solutions<0] = -1
        return self.planner.validateSolutions(solutions.astype(np.double), int(soc))
    

        