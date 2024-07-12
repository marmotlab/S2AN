import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import logging
import argparse
from map_generator import  dump_solutions
from runner import  RunnerBase as RunnerNoRay
from runner import Runner
from alg_parameters import *
import ray
import time
logging.basicConfig(level=logging.DEBUG)
from train_ray import test_through_benchmark
import numpy as np
if not ray.is_initialized():
    ray.init(num_gpus=SetupParameters.num_GPU)

# python test.py --obs 20 -o results/rl/20obs.csv 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_fname", '-o',help="statistic result")
    parser.add_argument("--obs", type=int, help="obstacle rate", default=20)
    parser.add_argument("--outputPaths", help="schedule for agents")
    parser.add_argument("-m", "--model", help="test trained model", default="0official.pth")

    args = parser.parse_args()
    testing_num_environments = 8  # ! reduce it if you run out of memory.

    outfname = args.output_fname
    solution_fname = args.outputPaths


    local_device = torch.device("cpu")
    # sampler = RunnerNoRay(0, local_device, is_sampler=True) 
    import os
    file_path = os.path.dirname(__file__)
    checkpoint_path = os.path.join(file_path,  "PPO_preTrained", "MapfSippsEnv", args.model)
    checkpoint = torch.load(checkpoint_path)
    
    TrainParam.N_ENVS = testing_num_environments
    sampler_refs = [Runner.remote(i + 1, local_device, is_sampler=True) 
        for i in range(TrainParam.N_ENVS)]
    
    output_path = os.path.dirname(outfname)
    os.makedirs(output_path, exist_ok=True )
    
    with open(outfname, 'a') as f:
        # num_agents, success_rate, calc_time_mean
        record = "num_agents, success_rate, reach_rate, collision_rate, "
        record += "calc_time_mean, calc_time_std, calc_time_max, "
        record += "makespan_mean, makespan_std, makespan_max"
        f.write(record +'\n')

    for num_agents in range(20,280,20):
        reward_test, sucessful, makespan, calc_time, reach_agents, num_collision_agents = \
            test_through_benchmark(sampler_refs, checkpoint, 
                                   num_agents, args.obs)
        # buffer = sampler.test(checkpoint, map_fnames, scen_fnames, num_agents)
        success_rate = sum(sucessful)/100
        calc_time = np.array(calc_time)
        makespan = np.array(makespan)
        sucessful = np.array(sucessful)
        reach_agents = np.array(reach_agents)
        collision_rates = ( np.array(num_collision_agents)).mean()/num_agents
        # calc_time = calc_time[sucessful]
        makespan = makespan[sucessful]
        reach_rate = reach_agents.mean()/num_agents
        print('num agents:', num_agents)
        print('calc time mean:', calc_time.mean())
        print('calc time std:', calc_time.std())
        print('calc time max:', max(calc_time))
        print('reward test:', reward_test)
        print('success rate:', success_rate)
        print('reach rate:', reach_rate)
        print('collision_rates:', collision_rates)
        print('makespan mean:', makespan.mean())
        print('makespan std:', makespan.std())
        print('makespan max:', max(makespan))

        
        with open(outfname, 'a') as f:
            # num_agents, success_rate, calc_time_mean
            record = "%d, %.4f, %.4f, %.4f, %.2f, %.4f, %.4f, %.2f, %.4f, %.4f"%(
                num_agents, success_rate, reach_rate, collision_rates
                , calc_time.mean(), calc_time.std(), calc_time.max()
                , makespan.mean(), makespan.std(), makespan.max()
                )
            f.write(record +'\n')

    
