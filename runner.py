import ray
import logging
logging.basicConfig(level=logging.ERROR)
import time

# from gym_priority import MapfPriorityEnv
# from gym_collision import MapfCollisionSearchEnv
from gym_sipps import MapfSippsEnv
from utils.utils import set_global_seeds
# from alg_parameters import *
from alg_parameters import *

# from PPO import PPO, RolloutBuffer
# from PPO_collision import PPO, RolloutBuffer
# from PPO_sipps import PPO, RolloutBuffer
from learning_sipps import PPO
from rollout_buffer import RolloutBuffer
import numpy as np
import torch
import copy
from alg_parameters import *
from rollout_buffer import EpisodeTensor, StateTensor

class RunnerBase:
    def __init__(self, env_id, device, is_sampler=True, env="MapfPriorityEnv",
                 writer=None):
        self.id = env_id
        set_global_seeds(env_id*123)
        
        # update policy for K epochs in one PPO update
        K_epochs = TrainParam.UPDATE_EPOCH
        gamma = TrainParam.GAMMA           # discount factor
        lr = TrainParam.lr      # learning rate for  network
        eps_clip = TrainParam.EPS_CLIP

        self.env = MapfSippsEnv()

        self.ppo_agent = PPO(lr, gamma, K_epochs, eps_clip, device, 
                             is_sampler=is_sampler, writer=writer)
     
    def sample(self, state_dict, device_to):
        """sample the buffer given the net_weights

        Args:
            state_dict (_type_): the given net_weights.
            device_to (torch.device): device to the problem.

        Returns:
            _type_: _description_
        """
        self.ppo_agent.load_from_state_dict(state_dict)
        from alg_parameters import TrainParam as TParams

        buffer = RolloutBuffer()
        # sample loop
        time_step = 0
        while time_step < TParams.N_STEPS:

            state = self.env.reset()
            self.ppo_agent.reset_agent()
            current_ep_reward = 0

            for _ in range(TParams.max_steps):

                # select action with policy
                action = self.ppo_agent.select_action(state, buffer, device_to)
                state, reward, done, _ = self.env.step(action)

                # saving reward and is_terminals
                buffer.rewards.append(torch.tensor(reward, dtype=torch.float32))
                buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # break; if the episode is over
                if done:
                    break

        buffer.is_terminals[-1] = True
        logging.info("sample done.")
        return buffer
    
    

    def train(self, buffer, buffer_demo, i_episode, device_to):
        self.ppo_agent.update(buffer, buffer_demo, i_episode)
        return self.get_model_state_dict(device_to)


    def test(self, state_dict, map_fnames, scen_fnames, num_agents):
        buffer =  RolloutBuffer()
        t_begin = time.time()
        self.ppo_agent.load_from_state_dict(state_dict)
        for map_fname, scen_fname in zip(map_fnames, scen_fnames):
            state = self.env.reset_by_testcase(map_fname, scen_fname, num_agents)
            
            from alg_parameters import TrainParam as TParams

            time_step = 0
            self.ppo_agent.reset_agent()
            current_ep_reward = 0
            device_to = self.ppo_agent.device
            inference = True
            for _ in range(TParams.max_steps):
                action = self.ppo_agent.select_action(
                    state, buffer, device_to, inference)
                state, reward, done, _ = self.env.step(action)

                # saving reward and is_terminals
                buffer.rewards.append(torch.tensor(reward, dtype=torch.float32))
                buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # break; if the episode is over
                if done:
                    t_now = time.time()
                    print("ray subdataset testing time:", t_now-t_begin)
                    buffer.calc_time.append(t_now-t_begin)
                    t_begin = t_now
                    buffer.makespan.append(state.solutions.shape[1])
                    buffer.num_collision_agents.append(state.num_collision_agents)
                    break
            
            # solutions = state.solutions
            # num_collision = state.num_collision_pairs
            buffer.is_terminals[-1] = True
            logging.info("test sampling done.")

        return buffer
    
    def get_model_state_dict(self, device):
        state_dict = self.ppo_agent.policy.state_dict()
        for p_name in state_dict:
            state_dict[p_name] = state_dict[p_name].to(device)
        return state_dict


# @ray.remote(num_cpus = 1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 2))
@ray.remote(num_cpus = 1, num_gpus=0)
class Runner(RunnerBase):
    def __init__(self, env_id, device, is_sampler=True, env="MapfPriorityEnv"):
        super().__init__(env_id, device, is_sampler, env)


