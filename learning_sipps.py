import torch
import torch.nn as nn
import numpy as np
from torch.distributions import  Categorical

from typing import List
from gym_sipps import State

from nets.attention_model import AttentionModel
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from alg_parameters import *
from rollout_buffer import *
from utils.utils import clip_grad_norms


class PPO:
    def __init__(self, lr, gamma, K_epochs, eps_clip, device, 
                 embedding_dim = NetParameters.EMBEDDING_DIM, 
                 is_sampler=True, writer=None): 

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        from alg_parameters import NetParameters as p
        self.policy = AttentionModel(embedding_dim,
                                     n_heads=p.N_HEAD,
                                     n_encode_layers=p.N_ENCODER_LAYERS,
                                     checkpoint_encoder=p.checkpoint_encoder,
                                     tanh_clip= TrainParam.CRITIC_MAX_VALUE,
                                     device=device).to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters() , lr)

        self.is_sampler = is_sampler
        self.MseLoss = nn.MSELoss()

        # tensorboard summary writer
        self.writer:SummaryWriter = writer


    def reset_agent(self):
        self.policy.reset()


    def select_action(self, state:State, buffer:RolloutBuffer, device_to, inference=False):
        assert(self.is_sampler)  # this function is used for sampling
        self.policy.eval()
        
        with torch.no_grad():
            state_tensor = StateTensor(state)
            
            episode_tensor = EpisodeTensor(state)
            if self.policy.fixed is None:
                # episode start. encode memory and record the static info.
                buffer.episode_tensors.append(episode_tensor)
                self.policy.encode(EpisodeTensors([episode_tensor]))

            batch_size = 1
            action_probs, state_val = self.policy.decode(
                episode_tensor, [state_tensor], state.num_agents, batch_size)
            
            # 1 action，n probabilities.
            dist = Categorical(action_probs.squeeze_())
            # action return int not one-hot vector. return index.
            action = dist.sample() if not inference else torch.argmax(action_probs)
            action_logprob = dist.log_prob(action)
        
        # careful! this will use lots of gpu memory
        buffer.states.append(StateTensor(state))
        buffer.actions.append(action)
        buffer.logprobs.append(action_logprob)
        buffer.state_values.append(state_val.squeeze_())

        return action.tolist()

    def update(self, buffer:RolloutBuffer, buffer_demo:RolloutBuffer, i_episode):
        assert(not self.is_sampler)  # this function is used for training
        self.policy.train()
        
        buffer_tensors = translate2tensor(buffer, self.gamma, self.device,
                                          normalized_return=False)

        # Optimize policy for K epochs
        for i_learning in range(self.K_epochs):
            i_iter = self.K_epochs*i_episode + i_learning
            rl_loss = self.calc_rl_loss(buffer_tensors, i_iter)

            loss = TrainParam.rl_weight *rl_loss 
            self.writer.add_scalar('Loss/loss', loss, i_iter)
            
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norms(self.optimizer.param_groups, TrainParam.max_grad_norm)
            self.optimizer.step()



    
    def calc_rl_loss(self, buffer_tensors,  i_iter):
        returns, rewards, old_actions, old_logprobs, old_state_values, \
            advantages, episode_tensors, state_tensors = buffer_tensors


        action_probs, state_vals = self.policy(episode_tensors, state_tensors)
        
        # 1个动作，n个可能的概率值
        dist = Categorical(action_probs) 
        
        # action return int not one-hot vector. return index.
        logprobs = dist.log_prob(old_actions)
        dist_entropy = dist.entropy()

        if TrainParam.ALGORITHM == "reinforce":
            policy_loss = -(returns * logprobs).mean()
            entropy_loss = - dist_entropy.mean() * TrainParam.entropy_weight
            loss = policy_loss + entropy_loss
            self.writer.add_scalar("Loss/policy_loss", policy_loss, i_iter)
            self.writer.add_scalar("Loss/returns", returns.mean(), i_iter)
            self.writer.add_scalar("Loss/entropy_loss", entropy_loss, i_iter)

        elif TrainParam.ALGORITHM == "ppo":

            # match state_values tensor dimensions with rewards tensor
            state_vals = torch.squeeze(state_vals)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss =  -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_vals, returns).mean()
            # TD loss
            # b = rewards.shape[0]
            # critic_loss = self.MseLoss(
            #     state_vals[...,:-1] - self.gamma* state_vals[...,1:] , 
            #     rewards[...,:-1]).mean() * ((b-1)/b)+\
            #     self.MseLoss(state_vals[...,-1], rewards[...,-1])/b

            entropy_loss = -dist_entropy.mean()
            loss = policy_loss + 0.5 *critic_loss +  0.01 * entropy_loss
            
            self.writer.add_scalar('Loss/policy_loss', policy_loss, i_iter)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, i_iter)
            self.writer.add_scalar('Loss/entropy_loss', entropy_loss, i_iter)
            self.writer.add_scalar('Training/KL_ratio', ratios.mean(), i_iter)
        
        else:
            assert False, "unsupported rl algorithm"
                        
        self.writer.add_scalar('Loss/rl_loss', loss, i_iter)
        return loss

    

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def load_from_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict)

