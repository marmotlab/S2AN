import ray
import torch
import logging
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
logging.basicConfig(level=logging.ERROR)
from datetime import datetime

from alg_parameters import *
from utils import get_log_fname, get_checkpoint_path
from runner import Runner
from runner import  RunnerBase as RunnerNoRay
import random
from typing import List
from rollout_buffer import concate_buffers
import copy

ray.init(num_gpus=SetupParameters.NUM_GPU)
run_num_pretrained = 0

writer = SummaryWriter(
    filename_suffix='_pretrainNum'+str(run_num_pretrained)
        +'_algo_'+TrainParam.ALGORITHM 
        +'_20obs_'+ str(EnvParam.N_AGENTS) + 'agents' 
        +'_worldSize'+str(EnvParam.WORLD_SIZE[0])
        +'_entropyWeight'+str(TrainParam.entropy_weight)
        # + '_debug'
)


def train(debug_mode = False):
    logging.info("training started.")
    logging.info(str(all_args))
    start_time = datetime.now().replace(microsecond=0)

    global_device = torch.device('cuda') \
        if SetupParameters.USE_GPU_TRAIN else torch.device('cpu')
    local_device = torch.device('cuda') \
        if SetupParameters.USE_GPU_SAMPLE else torch.device('cpu')
    
    num_envs = TrainParam.N_ENVS
    buffer_size = num_envs * TrainParam.N_STEPS
    if debug_mode:
        samplers = [RunnerNoRay(i + 1, local_device, is_sampler=True) 
            for i in range(num_envs)]
    else:
        sampler_refs = [Runner.remote(i + 1, local_device, is_sampler=True) 
                for i in range(num_envs)]

    trainer = RunnerNoRay(num_envs + 1, global_device, 
                                is_sampler=False, writer=writer)
    
    epoch = 0
    
    if TrainParam.load_pretrain and TrainParam.pretrain_path:
        checkpoint = torch.load(TrainParam.pretrain_path)
        trainer.ppo_agent.load_from_state_dict(checkpoint)
    
    net_weight = trainer.get_model_state_dict(local_device)

    log_n_episode = 0
    log_rewards = 0
    i_episode = 0
    time_step = 0
    
    buffer_demo = None

    if not debug_mode:
        test_reward, success_episodes, calc_time,_,_,_ = test_through_benchmark(sampler_refs, net_weight, EnvParam.N_AGENTS)
        writer.add_scalar("Test/reward", test_reward, i_episode)
        
    checkpoint_path = get_checkpoint_path(run_num_pretrained)
    pbar = tqdm(total=TrainParam.max_steps)
    while time_step <TrainParam.max_steps:
        epoch += 1
        t1 = time.time()
        

        # sample the buffers
        if  debug_mode:
            buffers = [sampler.sample(net_weight, local_device)
                    for sampler in samplers]
        else:
            net_weight_ref = ray.put(net_weight)

            buffer_refs = [sampler.sample.remote(net_weight_ref, local_device)
                        for sampler in sampler_refs]
            
            buffer_ready_refs, buffer_remain_refs = \
                    ray.wait(buffer_refs, num_returns=num_envs)
            assert(len(buffer_remain_refs)==0)
            buffers = ray.get(buffer_ready_refs)
            
        buffer = concate_buffers(buffers)
        buffer_size = len(buffer.actions)
        logging.info("sample rl buffer done.")

        buffer_episode = sum(buffer.is_terminals)
        logging.info("buffer size" +  str(buffer_size))

        logging.info('episodes: ' + str(buffer_episode))

        log_rewards += sum(buffer.rewards)

        log_n_episode += buffer_episode
        i_episode += buffer_episode
        pbar.update(buffer_size)

        t2 = time.time()
        logging.info('time sample: ' + str(t2-t1))

        # torch.autograd.set_detect_anomaly(True)
        # training        

        logging.info("training begin.")
        net_weight = trainer.train(buffer, buffer_demo, i_episode, local_device)
        buffer.clear()  # update rl buffer every time
        # buffer_demo clear when it update
        
        t3 = time.time()
        logging.info('time train: ' +str(t3-t2))

        if time_step % TrainParam.LOG_PERIOD == 0:
            log_avg_reward = log_rewards/log_n_episode
            
            writer.add_scalar("Training/reward", log_avg_reward, i_episode)
            log_rewards = 0
            log_n_episode = 0
            
            if not debug_mode:
                test_reward, success_episodes, calc_time,_,_,_ = test_through_benchmark(sampler_refs, net_weight, EnvParam.N_AGENTS)
                writer.add_scalar("Test/reward", test_reward, i_episode)

        if time_step % TrainParam.SAVE_PERIOD == 0:
            torch.save(net_weight, checkpoint_path)
        
        time_step += buffer_size

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")



def test_through_benchmark(samplers:List[Runner], net_weight, num_agents, obs=20):
    obs = int(obs)

    if EnvParam.WORLD_SIZE[0] == 32:
        file_path = "benchmark/random-32-32-"+ str(obs)+ "/"
        map_fnames = [file_path+"random-32-32-"+ str(obs)+ "-"+str(i)+".map" for i in range(100)]
        scen_fnames = [file_path+"random-32-32-"+ str(obs)+ "-"+str(i)+".scen" for i in range(100)]
    elif EnvParam.WORLD_SIZE[0] == 10:
        file_path = "benchmark/random-10-10-"+ str(obs)+ "/"
        map_fnames = [file_path+"random-10-10-"+ str(obs)+ "-"+str(i)+".map" for i in range(100)]
        scen_fnames = [file_path+"random-10-10-"+ str(obs)+ "-"+str(i)+".scen" for i in range(100)]
    elif EnvParam.WORLD_SIZE[0] == 64:
        file_path = "benchmark/random-64-64-"+ str(obs)+ "/"
        map_fnames = [file_path+"random-64-64-"+ str(obs)+ "-"+str(i)+".map" for i in range(100)]
        scen_fnames = [file_path+"random-64-64-"+ str(obs)+ "-"+str(i)+".scen" for i in range(100)]
    
    else:
        assert False, "unsupported world size for test. generate the benchmark first"
    n_envs = TrainParam.N_ENVS
    n_file = 100 // n_envs if 100%n_envs==0 else 100 // n_envs + 1
    
    index = [ i for i in range(100)]
    index = index[::n_file]
    index.append(100)
    
    net_weight_ref = ray.put(net_weight)
    
    buffer_refs =[ samplers[i].test.remote(net_weight_ref, 
        map_fnames[index[i]: index[i+1]], scen_fnames[index[i]: index[i+1]], 
        num_agents) for i in range(len(index) -1 )]
    
    buffer_ready_refs, buffer_remain_refs = \
        ray.wait(buffer_refs, num_returns=len(index) -1)
    assert(len(buffer_remain_refs)==0)
    buffers = ray.get(buffer_ready_refs)
    buffer = concate_buffers(buffers)
    
    # sum up the rewards
    reward_test = sum(buffer.rewards)/sum(buffer.is_terminals)
    total_r = 0
    success_episodes = 0
    sucessfuls = []
    reach_agents = []
    reach_agent = 0 
    for r, is_termial in zip(buffer.rewards, buffer.is_terminals):
        total_r += r
        if r>=0: # if have agent collision. this agent failed.
            reach_agent += 1
        if is_termial:
            if total_r == 20:
                success_episodes += 1
                sucessfuls.append(True)
            else:
                sucessfuls.append(False)
            reach_agents.append(reach_agent)
            total_r = 0
            reach_agent = 0
    calc_time = copy.copy(buffer.calc_time)
    makespan = copy.copy(buffer.makespan)
    num_collision_agents = copy.copy(buffer.num_collision_agents)
    buffer.clear()
    ray.internal.free(net_weight_ref)
    return reward_test, sucessfuls, makespan, calc_time, reach_agents, num_collision_agents



if __name__ == '__main__':
    # debug_mode = True
    debug_mode = False
    train(debug_mode)

    