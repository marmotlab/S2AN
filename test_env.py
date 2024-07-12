from gym_sipps import MapfSippsEnv
import numpy as np
import logging
import os
from map_generator import MapLoader
from gym_sipps import State

logging.basicConfig(level=logging.DEBUG)

def test_sipps_env():
    print('start testing step function of SIPPS env.')

    np.random.seed(10)
    env = MapfSippsEnv()
    
    num_agents = env.num_agents
    logging.info('number of agents: ' + str(num_agents))

    seq = np.arange(num_agents)
    np.random.shuffle(seq)

    print("start randomly to pick up agents.")
    for i in range(num_agents):
        observation, reward, done, info =env.step(int(seq[i]))
        print('iter', i, '. Choose agent', seq[i],  '\t Done?', done, '. Reward: ', reward, 'info:', info)
        if done or done is None:
            break
    print('Test SIPPS env through.')
    return True


def test_specific_scen():
    file_path = "/media/tiecun/Data/2_keyan/0codes/PriorityPathFinding"
    map_fname = os.path.join(file_path, 
                             "benchmark", "12_09_2023-14_13_21_894871.map")
    scen_fname = os.path.join(
        file_path, "benchmark", "12_09_2023-14_13_21_894871.scen")

    num_agents = 100 # max 409 in random map
    loader = MapLoader()

    obstacles = loader.read_map(map_fname)
    start_locs, goal_locs = loader.read_scen(scen_fname, num_agents)
    world, goals = loader.assemble_input(obstacles, start_locs, goal_locs)
    
    state = State(world, goals, num_agents)
    target = state.target_matrix()
    
    state.transit()
    print('hold on')

if __name__ == '__main__':
    test_sipps_env()
    # test_specific_scen()


