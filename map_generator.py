import logging
import numpy as np
import random
import sys
import random
import time
from datetime import  datetime

from alg_parameters import *
import os

# global value of this file
map_generator_fpath = os.path.dirname(__file__)

# Map, agent start locations, goal locations.
class Instance():
    def __init__(self, num_agents, map_wid, map_height, obstacle_prob):
        logging.info("instance initialize.")
        self.num_agents = num_agents

        self.map_wid = int(map_wid)
        self.map_height = int(map_height)
        self.map_size = self.map_wid * self.map_height
        self.num_obstacles = int(self.map_size * obstacle_prob)
        self.obstacle_prob = obstacle_prob
   
    def generate_instance(self):
        """randomly generate obstacles and find connected start and goal locations. the same as in PRIMAL.

        Returns:
           world:  np.ndarray. [world_wid, world_height]. -1: obstacle. 0:free space. >0: agent i-1's start location.
           goals: np.ndarray. [world_wid, world_height]. 0: no meaning. >0: agent i-1's goal location.
        """
        map_wid = self.map_wid
        obstacle_prob = self.obstacle_prob
        # randomly generate obstacles.
        world = -(np.random.rand(int(map_wid), int(map_wid)) < obstacle_prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
        self.world_shape = world.shape
        
        start_locations = self.add_start_locations(world)
        goals = self.generate_goal_locations(world, start_locations)

        return world, goals

    def generate_connected_instance(self):
        """generate the environment as in Priority Based Search. the world is single connected. It's harder to find a feasible solutions.

        Returns:
           world:  np.ndarray. [world_wid, world_height]. -1: obstacle. 0:free space. >0: agent i-1's start location.
           goals: np.ndarray. [world_wid, world_height]. 0: no meaning. >0: agent i-1's goal location.
        """
        logging.info("gen obstacle:")
        t1 = time.time()
        self.generate_connected_map(self.map_wid, self.map_height, self.num_obstacles)
        world = self.obstacle_map
        t2 = time.time()
        logging.info("gen obstacle time: "+str(t2-t1))
        logging.info("add start locations")
        self.start_locs = self.add_start_locations(world)
        t3 = time.time()
        logging.info("start location time: "+ str(t3-t2))
        goals, self.goal_locs = self.generate_goal_locations_in_connected(world)
        t4 = time.time()
        logging.info("goals location time: "+ str(t4-t3))

        return world, goals

    def generate_instance_by_map(self, fname_map, fname_scene, num_agents=50):

        data_loader = MapLoader()
        world, goals = data_loader.load_movingai(fname_scene, fname_map, num_agents)        
        return world, goals

    # belows are lowlevel interface. use carefully.
    def generate_connected_map(self,  rows, cols, obstacles):
        """_summary_

        Args:
            rows (_type_): _description_
            cols (_type_): _description_
            obstacles (_type_): _description_
        """
        self.map_wid = rows + 2
        self.map_height = cols + 2
        self.map_size = self.map_wid * self.map_height
        # self.my_map.resize(map_size, False)
        self.obstacle_map = np.zeros([self.map_wid, self.map_height], dtype=int)

        # add padding? why do i need to extend the map?
        self.obstacle_map[0, :] = -1
        self.obstacle_map[-1, :] = -1
        self.obstacle_map[:, 0] = -1
        self.obstacle_map[:, -1] = -1

        # add obstacles uniformly at random
        t1 = time.time()
        i = 0
        while i < obstacles:        
            loc = random.randint(0,self.map_size-1)
            loc_index = self.get_index(loc)
            if self.add_obstacle(loc_index):
                i+=1
        t2 = time.time()
        logging.info("map time: " +str(t2-t1))
        # delete the padding
        self.obstacle_map = self.obstacle_map[1:-1, 1:-1]
        self.map_wid, self.map_height = self.obstacle_map.shape
        self.map_size = self.map_wid * self.map_height 
    
    def get_index(self, id):
        row = id // self.map_height
        col = id % self.map_height
        return row, col

    def add_obstacle(self, loc)->bool:
        if self.obstacle_map[loc] == -1:
            return False
        self.obstacle_map[loc] = -1
        obstacle_x = loc[0]
        obstacle_y = loc[1]
        x = [obstacle_x, obstacle_x + 1, obstacle_x, obstacle_x - 1 ]
        y = [obstacle_y - 1, obstacle_y, obstacle_y + 1, obstacle_y ]
        start = 0
        goal = 1
        while (start < 3 and goal < 4):
        
            if x[start] < 0 or x[start] >= self.map_wid or y[start] < 0 or y[start] >= self.map_height \
                or self.obstacle_map[x[start], y[start]]:
                start+=1
            elif goal <= start:
                goal = start + 1
            elif x[goal] < 0 or x[goal] >= self.map_wid or y[goal] < 0 or y[goal] >= self.map_height \
                or self.obstacle_map[x[goal], y[goal]]:
                goal+=1
            elif self.isConnected([x[start], y[start]], [x[goal], y[goal]]): # cannot find a path from start to goal 
            
                start = goal
                goal+=1
            
            else:            
                self.obstacle_map[loc] = 0
                return False
        
        return True

    def get_valid_neighbors(self, position):
        wid, height = self.map_wid, self.map_height
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        neighbors = []
        for i in range(4):
            x = position[0] + dx[i]
            y = position[1] + dy[i]
            if 0 <= x and x<=wid-1 and 0<=y and y<=height-1\
                and self.obstacle_map[x][y] >-1: # -1 means obstacle
                neighbors.append([x,y])
        return neighbors
    
    def isConnected(self, start, goal):    
        open_queue = []
        closed = np.zeros([self.map_wid, self.map_height], dtype = bool)
        open_queue.append(start)
        closed[start[0], start[1]] = True
        while len(open_queue) > 0:
            assert(closed.sum() < self.map_size)
            curr = open_queue.pop(0)
            if (curr == goal):
                return True
            for  next in self.get_valid_neighbors(curr):            
                if closed[next[0], next[1]]:
                    continue
                open_queue.append(next)
                closed[next[0], next[1]] = True
        
        return False
    
    def add_start_locations(self, world):
        # world. donot copy world. will add the start location to world.
        # randomize the position of agents
        start_counter = 1
        start_locations = []
        while start_counter <= self.num_agents:
            x, y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
            if world[x, y] == 0:
                world[x, y] = start_counter
                start_locations.append((x, y))
                start_counter += 1
        return start_locations

    def generate_goal_locations_in_connected(self, world):
        goal_counter = 1
        goal_locations = []
        goals = np.zeros_like(world)
        while goal_counter <= self.num_agents:
            x, y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
            if world[x, y] > -1 and goals[x, y] == 0:
                goals[x, y] = goal_counter
                goal_locations.append((x, y))
                goal_counter += 1
        return goals, goal_locations
    
    def generate_goal_locations(self, world, start_locations):
        # randomize the position of goals
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions = dict()
        while goal_counter <= self.num_agents:
            agent_pos = start_locations[goal_counter - 1]
            valid_tiles = self.get_connected_region(world, agent_regions, agent_pos[0], agent_pos[1])
            x, y = random.choice(list(valid_tiles))
            if goals[x, y] == 0 and world[x, y] != -1:
                # ensure new goal does not at the same grid of old goals or obstacles
                goals[x, y] = goal_counter
                goal_counter += 1
        return goals
    
    def get_connected_region(self, world0, regions_dict, x0, y0):
        # ensure at the beginning of an episode, all agents and their goal at the same connected region
        sys.setrecursionlimit(1000000)
        if (x0, y0) in regions_dict:  # have done
            return regions_dict[(x0, y0)]
        visited = set()
        sx, sy = world0.shape[0], world0.shape[1]
        work_list = [(x0, y0)]
        while len(work_list) > 0:
            (i, j) = work_list.pop()
            if i < 0 or i >= sx or j < 0 or j >= sy:
                continue
            if world0[i, j] == -1:
                continue  # crashes
            if world0[i, j] > 0:
                regions_dict[(i, j)] = visited
            if (i, j) in visited:
                continue
            visited.add((i, j))
            work_list.append((i + 1, j))
            work_list.append((i, j + 1))
            work_list.append((i - 1, j))
            work_list.append((i, j - 1))
        regions_dict[(x0, y0)] = visited
        return visited
    

# load the map from the moving ai scene
class MapLoader():
    def __init__(self):
        pass

    def load_movingai(self, scen_fname, map_fname, num_agents=50):
        """ return world, goals
        """
        obstacles = self.read_map(map_fname)
        start_locs, goal_locs = self.read_scen(scen_fname, num_agents)
        world, goals = self.assemble_input(obstacles, start_locs, goal_locs)
        return world, goals
    
    def dump_movingai(self, world:np.ndarray, goals:np.ndarray, 
                      path=None, fname=None, num_agents=-1):
        print("saving moving ai scen and map")

        if path is None:
            path = os.path.join(map_generator_fpath, "benchmark")
        if fname is None:
            fname = generate_timestamp_prefix()

        fname_with_path = os.path.join(path, fname)
        map_fname = fname_with_path + ".map"
        scene_fname = fname_with_path +".scen"
        
        map = world.copy().astype(int)
        map[map>0]  = 0
        wid, height = world.shape
        with open(map_fname, "w") as f:
            # save map  
            f.write("type octile\n")
            f.write("height " + str(world.shape[0]) + "\n")
            f.write("width " + str(world.shape[1]) + "\n")
            f.write("map\n")
            for i in range(wid):
                map_string = ['@' if map[i, j]<0 else '.' for j in range(height)]
                map_string = ''.join(map_string)
                f.write(map_string+'\n')

        if num_agents == -1:
            num_agents = goals.max() 

        goal_locations = self.find_locations(goals, num_agents)
        start_locations = self.find_locations(world, num_agents)
        
        # save scene
        map_fname_wo_path = fname + '.map'
        with open(scene_fname, 'w') as f:
            f.write('version 1\n')
            for i in range(num_agents):
                # y x y x
                f.write(f'{i}\t{map_fname_wo_path}\t{wid}\t{height}\t')
                f.write(f'{start_locations[i][1]}\t{start_locations[i][0]}\t')
                f.write(f'{goal_locations[i][1]}\t{goal_locations[i][0]}\t')
                f.write(f'-1\n')

    # below are lowlevel interface. use carefully.
    # 读写随机提供的地图文件，并转换成输入
    def read_map(self, fname):
        world_len =  world_wid = 0
        
        obstacles = -1
        with open(fname, 'r') as fp:
            for i in range(4):
                l = fp.readline()
                if i == 1:
                    words = l.strip().split()
                    world_len = int(words[-1])
                if i== 2:
                    words = l.strip().split()
                    world_wid = int(words[-1])
            
            obstacles = np.zeros([world_len, world_wid])
            row = 0
            while True:
                line = fp.readline()
                if not line:
                    break
                for j in range(world_wid):
                    if line[j] != '.':
                        obstacles[row, j] = -1
                row += 1

        return obstacles

    # 读写随机提供的场景文件，并转换成输入
    def read_scen(self, fname, num_agents):
        start_locations = np.zeros([num_agents, 2], dtype=int)
        goal_locations = np.zeros([num_agents, 2], dtype=int)
        with open(fname, 'r')  as fp:
            fp.readline()
            for i in range(num_agents):
                line = fp.readline()
                if len(line) == 0 :
                    logging.error("Error! The instance has only " , i , " agents.")
                    break
                words = line.split()
                start_locations[i, 0] = int(words[5])
                start_locations[i, 1] = int(words[4])
                goal_locations[i, 0] = int(words[7])
                goal_locations[i, 1] = int(words[6])
        return start_locations, goal_locations

    def assemble_input(self, obstacles, start_locations, goal_locations):
        world_len, world_wid = obstacles.shape
        num_agents = start_locations.shape[0]
        goals = np.zeros([world_len, world_wid])
        world = obstacles  # shallow copy.
        for i in range(num_agents):
            world[int(start_locations[i, 0]), int(start_locations[i, 1])] = i + 1
            goals[int(goal_locations[i, 0]), int(goal_locations[i, 1])] = i + 1
        return world, goals

    def find_locations(self, grid_world:np.ndarray, num_agents=-1):
        locations = np.where(grid_world>0)
        if (num_agents == -1):
            num_agents = grid_world.max() - 1
        sorted_locations = np.zeros([num_agents, 2]).astype(int)

        for i in range(num_agents):
            index0 = locations[0][i]
            index1 = locations[1][i]
            id = int(grid_world[index0, index1] - 1)
            sorted_locations[id, :] = [index0, index1]
        return sorted_locations.astype(int)


def dump_solutions(solutions, path=None, fname=None):
    print("saving solutions.")
    if path is None:
        path = os.path.join(map_generator_fpath, "results")
    if not os.path.exists(path):
        os.makedirs(path)
    if fname is None:
        fname = generate_timestamp_prefix()

    result_fname = os.path.join(path, fname+"_schedule.yaml" )

    with open(result_fname, "w") as f:
        max_t = solutions.shape[1]
        num_agents = solutions.shape[0]
        f.write("schedule:\n")
        for i in range(num_agents):
            f.write("  agent" + str(i) + ":\n")
            for t in range(max_t):
                x ,y = solutions[i, t, :]
                if x<0:
                    break
                f.write("    - x: " + str(x) + "\n")
                f.write("      y: " + str(y) + "\n")
                f.write("      t: " + str(t) + "\n")        

def generate_timestamp_prefix():
    now = datetime.now()
    # add nanosecond
    time_now = time.time()
    time_micro = int((time_now - int(time_now))*1e6)
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
    dt_string += "_" + str(time_micro)
    return dt_string

config1 = {
    'map_wid' : 32, 'map_height' : 32, 'obstacle_rate': 20, 'max_agents': 400
}
config2 = {
    'map_wid' : 64, 'map_height' : 64, 'obstacle_rate': 20, 'max_agents': 800
}

def generate_benchmark(cfg: dict):

    map_prefix = 'random-'+str(cfg['map_wid'])+\
            '-'+str(cfg['map_height'])+'-'+str(int(cfg['obstacle_rate']))
    folder_path = os.path.join(map_generator_fpath, 'benchmark', map_prefix)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    loader = MapLoader()
    # generate 100 test cases with max agents.
    for i in range(100):
        instance = Instance(cfg['max_agents'], cfg['map_wid'],
                             cfg['map_height'], 0.01*cfg['obstacle_rate'])
        world, goals = instance.generate_connected_instance()
        loader.dump_movingai(world, goals, folder_path, map_prefix+'-'+str(i), 
                             cfg['max_agents'])
        
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # generate the test cases
    # generate_benchmark(config1)
    generate_benchmark(config2)

