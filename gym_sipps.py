import logging
from lowlevel_solvers.pysipps_planner import SippsPlanner as Planner
import gym
import math
import numpy as np
from queue import Queue
import time

from alg_parameters import *
from map_generator import Instance, MapLoader



# state: input information. use pypriority_planner to solve problem.
class State:
    def __init__(self, world, goals, num_agents, save_mode=False, screen=0) -> None:
        """initialization"""
        self.state = world.copy()  # static obstacle: -1,empty: 0,agent = positive integer (agent_id)
        self.goals = goals.copy()  # empty: 0, goal = positive integer (corresponding to agent_id)
        self.num_agents = num_agents
        self.save_mode = save_mode
        self.start_locations, self.goal_locations = self.scan_for_agents()  # position of agents, and position of goals

        assert (len(self.start_locations) == num_agents)
        self.obstacle_map = self.state.copy()
        self.obstacle_map[self.obstacle_map>0] = 0
        self.obstacle_map[self.obstacle_map<0] = 1
        # obstacle map: 1 obstacle. 0 free space.
        
        self.planner = Planner(world, goals, num_agents, screen = screen)
        self.target_matrix = self.planner.get_target_matrix()

        # self.features_map = self.get_map_raw_features()
        self.features_map = self.get_map_dense_features()

        self.agents_planned_id = []
        self.agents_planned_bool = [ False  for _ in range(num_agents)]

        self.num_collision_pairs = 0
        self.num_collision_agents = 0
        self.harmful = np.array(detect_harmful_goals(self.obstacle_map, self.goal_locations, self.num_agents))
        
        goal_sit_on_start = find_goal_sit_on_start(self.start_locations, self.goal_locations)

        self.harmful = np.logical_or(self.harmful, goal_sit_on_start)

        danger = self.target_matrix.sum(axis=0)>0
        # minimal a* would only go through the harmfuls and sit on starts.
        # assert np.all(self.harmful[danger]), "all the target should be harmful"
   
    def find_rest_no_collision(self):
        rest_collision = self.collision_edge_index[~np.array(self.agent_visited), :]
        if np.any(rest_collision.sum(axis=1)>0): # have collision
            return False
        else:
            return True
    
    def get_isdone(self):

        return len(self.agents_planned_id) == self.num_agents

    def scan_for_agents(self):
        """find the position of agents and goals"""
        start_locations = [(-1, -1) for _ in range(self.num_agents)]
        goal_locations = [(-1, -1) for _ in range(self.num_agents)]

        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):  # check every position in the environment
                if self.state[i, j] > 0:  # agent
                    start_locations[int(self.state[i, j]) - 1] = (i, j)
                if self.goals[i, j] > 0:  # goal
                    goal_locations[int(self.goals[i, j]) - 1] = (i, j)
       
        assert ((-1, -1) not in start_locations and (-1, -1) not in goal_locations)
        return np.array(start_locations), np.array(goal_locations)
    
    def get_map_features(self):
        wid, height = self.state.shape
        max_distance = wid*height
        features = np.zeros([self.num_agents, 3, wid, height])
        for i in range(self.num_agents):
            dmap_start = self.bfs(self.start_locations[i])
            dmap_goal = self.bfs(self.goal_locations[i])
            goal_pos = self.goal_locations[i]
            fmap = dmap_start + dmap_goal
            fmap[fmap<0] = max_distance * 2
            f_min = dmap_start[goal_pos[0], goal_pos[1]]
            fmap = f_min/fmap
            dmap_goal[dmap_goal<0] = 0
            dmap_start[dmap_start<0] = 0

            features[i, 0, :] = fmap
            features[i, 1, :] = dmap_goal/max_distance*5
            features[i, 2, :] = dmap_start/max_distance*5

        return features
    
    def get_map_raw_features(self):
        ''' get the raw feature. [num_agents, 3, wid, height]. dim 0-1 start location and goal
        one-hot map. dim 2: obstacle map.
        '''
        wid, height = self.state.shape
        features = np.zeros([self.num_agents, 3, wid, height])
        for i in range(self.num_agents):

            start_x, start_y = self.start_locations[i]
            goal_x, goal_y = self.goal_locations[i]

            features[i, 0, start_x, start_y] = 1
            features[i, 1, goal_x, goal_y] = 1
            features[i, 2, :] = self.obstacle_map

        return features

    def get_map_dense_features(self):
        ''' get the raw feature. [3, wid, height]. dim 0-1 start location and goal
        one-hot map. dim 2: obstacle map.
        '''
        wid, height = self.state.shape
        features = np.zeros([3, wid, height])
        for i in range(self.num_agents):

            start_x, start_y = self.start_locations[i]
            goal_x, goal_y = self.goal_locations[i]

            features[ 0, start_x, start_y] = 1
            features[ 1, goal_x, goal_y] = 1
        features[ 2, :] = self.obstacle_map

        return features

    def get_valid_neighbors(self, position):
        world = self.state
        wid, height = world.shape
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        neighbors = []
        for i in range(4):
            x = position[0] + dx[i]
            y = position[1] + dy[i]
            if 0 <= x and x<=wid-1 and 0<=y and y<=height-1\
                and world[x][y] >-1: # -1 means obstacle
                neighbors.append([x,y])
        return neighbors

    def bfs(self, begin):
        # need the obstacle and map size from self.
        distance_map = -1 * np.ones_like(self.state)
        q = Queue()
        begin_distance = 1 # not 0. 起点终点重合会有问题
        q.put((begin, begin_distance))
        distance_map[begin[0], begin[1]] = begin_distance
        world = self.state
        wid, height = world.shape
        max_d_now = wid*height
        num_iter = 0
        while not q.empty():
            p = q.get()
            cost_parent = p[1]
            cost_child = cost_parent + 1
            num_iter += 1
            neighbors = self.get_valid_neighbors(p[0])
            for n in neighbors:                
                distance_n = distance_map[n[0], n[1]]
                if distance_n == -1:
                    distance_map[n[0], n[1]] = cost_child
                    q.put((n, cost_child))
                elif cost_child < distance_n:
                    distance_map[n[0], n[1]] = cost_child
                    print('weird')
                    assert(False)
            if num_iter > max_d_now:
                print('weird. quit from max distance')
                assert(False)
                break
        return distance_map


    def transit(self, action):
        ''' action: agent id.
        '''
        assert not self.get_isdone(), "the state is done. cannot transit anymore."
        assert not self.agents_planned_bool[int(action)],  str(action) + " planned before."
        self.agents_planned_bool[action] = True
        self.agents_planned_id.append(action)


        self.planner.solveSingleAgent(action, self.agents_planned_id[:-1])
        self.solutions = self.planner.get_solutions()

        self.priority_graph = self.planner.get_priorities()
        self.collision_edge_index = self.planner.get_collision_edge_index()

        self.num_collision_pairs = self.collision_edge_index.shape[0]/2
        self.num_collision_agents = len(set(self.collision_edge_index[:, 0]))


class StateIter(State):
    def __init__(self, world, goals, num_agents, save_mode=False, screen=0) -> None:
        super().__init__(world, goals, num_agents, save_mode, screen)
        self.agent_seq = [i for i in range(num_agents)]
        self.planner.calcInitialSolution(self.agent_seq)
        self.n_iter = 0
    
    def transit(self, action):
        assert not self.get_isdone(), "the state is done. max iter number reach."
        # put one agent in the back
        action = int(action)
        self.agent_seq.remove(action)
        self.agent_seq.append(action)
        
        self.planner.step([action])
        self.n_iter+=1

        self.solutions = self.planner.get_solutions()

        self.priority_graph = self.planner.get_priorities()
        self.collision_edge_index = self.planner.get_collision_edge_index()

        self.num_collision_pairs = self.collision_edge_index.shape[0]/2
        self.num_collision_agents = len(set(self.collision_edge_index[:, 0]))
    
    def get_collision_edge_index(self):
        return self.planner.get_collision_edge_index()
    
    def get_isdone(self):
        
        return self.n_iter>= self.num_agents

def detect_harmful_goals(obstacles_map, goal_locs, num_agents):
    # obstacles_map: 1 obstacle. 0 empty. 
    obs_pro = -obstacles_map.copy()
    obs_pro[obs_pro>0] = 0     # start locations remove.
    goals = np.zeros_like(obs_pro)
    for i in range(num_agents):
        goals[tuple(goal_locs[i])] = -1
    # should I enlarge the obs pro?
    for i in range(num_agents):
        obs_pro[goal_locs[i][0], goal_locs[i][1]] = -1
    map_wid, map_height = obstacles_map.shape
    instance = Instance(num_agents, map_wid, map_height, 0.2)
    harmful = [False for i in range(num_agents)]
    x_neighbors = [0, 1, 0, -1]
    y_neighbors = [-1, 0, 1, 0]
    for i in range(num_agents):
        neighbors = []
        x, y = goal_locs[i]
        for j in range(4):
            x_nei = x_neighbors[j] + x
            y_nei = y_neighbors[j] + y
            if x_nei < 0 or x_nei >= goals.shape[0] or y_nei < 0 or y_nei>= goals.shape[1]:
                continue
            if goals[x_nei, y_nei] == -1:
                neighbors.append((x_nei, y_nei))

        obs_pro[goal_locs[i][0], goal_locs[i][1]] = 0
        instance.obstacle_map = obs_pro
        is_connected = instance.add_obstacle(tuple(goal_locs[i]))

        while is_connected and len(neighbors)>0:
            obs_pro[goal_locs[i][0], goal_locs[i][1]] = 0
            obs_pro[neighbors[0]] = 0
            is_connected = instance.add_obstacle(tuple(goal_locs[i]))
            obs_pro[neighbors[0]] = -1
            neighbors.pop(0)
            
        harmful[i] = not is_connected
        
        # make sure it did it.
        obs_pro[goal_locs[i][0], goal_locs[i][1]] = -1

    return harmful

def find_goal_sit_on_start(start_locs, goal_locs):
    num_agents = start_locs.shape[0]
    goal_sit_on_start = np.zeros([num_agents])
    for i in range(num_agents):
        for j in range(num_agents):
            if i==j:
                continue
            if np.all(goal_locs[i] == start_locs[j]):
                goal_sit_on_start[i] = 1
                break
    return goal_sit_on_start

class MapfSippsEnv(gym.Env):
    """construct MAPF problems to a standard RL environment"""
    def __init__(self, num_agents=EnvParam.N_AGENTS, size=EnvParam.WORLD_SIZE,
                 prob=EnvParam.OBSTACLE_PROB, state_class_name="State",save_mode=False, screen=0):
        """initialization"""
        logging.info("env initialize")
        self.num_agents = num_agents
        self.SIZE = size  # size of a side of the square grid
        self.PROB = prob  # obstacle density
        self.save_mode = save_mode
        self.screen = screen
        
        StateClass = {
            'State': State,
            'StateIter': StateIter
        }.get(state_class_name, None)
        assert StateClass is not None, "input invalid state class name"
        
        self.is_done = True
        while self.is_done:
            map_wid, obstacle_prob = self.sample_map_args()
            self.instance = Instance(num_agents, map_wid, map_wid, obstacle_prob)
            # world, goals = self.instance.generate_instance()
            world, goals = self.instance.generate_connected_instance()
            self.state = StateClass(world, goals, num_agents, 
                                    save_mode=save_mode, screen=screen)
            self.is_done = self.state.get_isdone()


    def sample_map_args(self):
        obstacle_prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1],
                        self.PROB[1])  # sample a value from triangular distribution
        map_wid = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]],
                                p=[.5, .25, .25])  # sample a value according to the given probability
        return map_wid, obstacle_prob

    def reset(self, num_agents=EnvParam.N_AGENTS):
        """restart a new task"""
        logging.info("\nenv reset")
        self.num_agents = num_agents

        self.is_done = True
        while self.is_done:
            map_wid, obstacle_prob = self.sample_map_args()
            self.instance = Instance(num_agents, map_wid, map_wid, obstacle_prob)            
            # world, goals = self.instance.generate_instance()  
            world, goals = self.instance.generate_connected_instance()
            self.state = State(world, goals, num_agents, self.save_mode, screen=self.screen)
            self.is_done = self.state.get_isdone()
        return self.state

    def reset_by_testcase(self, map_fname, scen_fname, num_agents):
        """restart a new task"""
        logging.info("\nenv reset by test file")
        loader = MapLoader()
        obstacles = loader.read_map(map_fname)
        start_locs, goal_locs = loader.read_scen(scen_fname, num_agents)
        world, goals = loader.assemble_input(obstacles, start_locs, goal_locs)
        world = world.astype(int)
        goals = goals.astype(int)
        self.num_agents = num_agents

        self.is_done = True
        while self.is_done:
            self.state = State(world, goals, num_agents, self.save_mode, screen=self.screen)
            self.is_done = self.state.get_isdone()
        return self.state


    def step(self, action):
        action = int(action)
        num_collisions_last = self.state.num_collision_pairs
        num_collision_agent_last = self.state.num_collision_agents
        try:
            # done: environment end. is_success: solve the problem.
            # someone cannot find the path due to the priority.
            self.state.transit(action)
            done = self.state.get_isdone()

            # 先学一版能够顺利无碰撞的. log(1+x). 否则方差太大不好学
            # 或者采取碰撞的智能体的数量？
            
            # reward = -math.log(self.state.num_collision_pairs - num_collisions_last + 1)
            reward = - (self.state.num_collision_agents - num_collision_agent_last)
            
            # add some reward if planned successfully.
            if done and self.state.num_collision_pairs == 0:
                reward = 20
            
        except:
            print('error')
            assert(False)

        return self.state, reward, done, None
