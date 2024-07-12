import torch
from gym_sipps import State
from typing import List



class EpisodeTensor:
    # store the static info in one episode
    def __init__(self, s:State) -> None:
        # starts, goals [n, 2]. 
        self.starts = torch.tensor(s.start_locations, dtype=torch.float)
        self.goals = torch.tensor(s.goal_locations, dtype=torch.float)
        # obs_map [3, map_wid, map_height]
        self.features_map = torch.tensor(s.features_map)        

        # TODO figure out transpose or not.
        # 不需要转置。因为若a被b堵住，说明 应该对b更关注。有意义
        self.target_matrix = torch.tensor(
            s.target_matrix, dtype=torch.float32)
        self.harmful = torch.tensor(
            s.harmful, dtype=torch.float32)
        
        self.num_agents = s.num_agents

    def detach(self):
        # 就地转换
        self.starts = self.starts.detach()
        self.goals = self.goals.detach()
        self.features_map = self.features_map.detach()
        self.target_matrix = self.target_matrix.detach()
        self.harmful = self.harmful.detach()
        return self
    
    def to(self, device):
        # 就地转换
        self.starts = self.starts.to(device)
        self.goals = self.goals.to(device)
        self.features_map = self.features_map.to(device)
        self.target_matrix = self.target_matrix.to(device)
        self.harmful = self.harmful.to(device)
        return self
    
class EpisodeTensors:
    def __init__(self, et:List[EpisodeTensor]) -> None:
                # starts, goals [n, 2]. 
        
        self.starts = torch.stack([e.starts for e in et])
        self.goals = torch.stack([e.goals for e in et])
        self.features_map = torch.stack([e.features_map for e in et])

        self.target_matrix = torch.stack([e.target_matrix for e in et])
        self.harmful = torch.stack([e.harmful for e in et])
        
        self.num_agents = et[0].num_agents
    
    def detach(self):
        # 就地转换
        self.starts = self.starts.detach()
        self.goals = self.goals.detach()
        self.features_map = self.features_map.detach()
        self.target_matrix = self.target_matrix.detach()
        self.harmful = self.harmful.detach()
        return self
    
    def to(self, device):
        # 就地转换. 节省内存
        self.starts = self.starts.to(device)
        self.goals = self.goals.to(device)
        self.features_map = self.features_map.to(device)
        self.target_matrix = self.target_matrix.to(device)
        self.harmful = self.harmful.to(device)
        return self
        

class StateTensor:
    def __init__(self, s:State=None):
       if s is not None:
            self.agents_planned = torch.tensor(
                s.agents_planned_id, dtype=torch.int64)
    
    def init_by_planned_ids(self, agents_planned_id):
        self.agents_planned = torch.tensor(
                agents_planned_id, dtype=torch.int64)
    
    def to(self, device):
        self.agents_planned = self.agents_planned.to(device)
        return self
    def detach(self):
        self.agents_planned = self.agents_planned.detach()
        return self

def concate_state_tensor(states:List[StateTensor], device):
    # it's hard to convert the states from list of class to tensor.
    # states = [Data(s) for s in self.states]
    
    # num_agents is different in different episode.
    features_map = torch.cat(
        [s.features_map for s in states], dim=0).to(device)
    
    num_agents_list = [s.num_agents for s in states]
    target_matrix = [s.target_matrix for s in states]
    harmful = [s.harmful for s in states]
    agents_planned =  torch.cat(
        [s.agents_planned for s in states], dim=0).to(device)
    

    return features_map, target_matrix, harmful, agents_planned


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states : List[StateTensor] = []
        self.episode_tensors: List[EpisodeTensor] = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

        # for testing validation
        self.calc_time = []
        self.makespan = []
        self.num_collision_agents =[]
    
    def to(device):
        
        pass
    
    def __iadd__(self, other):
        self.actions += other.actions
        self.states += other.states
        self.episode_tensors += other.episode_tensors
        self.logprobs += other.logprobs
        self.rewards += other.rewards
        self.state_values += other.state_values
        self.is_terminals += other.is_terminals
        self.calc_time += other.calc_time
        self.makespan += other.makespan
        self.num_collision_agents += other.num_collision_agents
        return self

    def cut_out(self, batch_size):
        self.actions = self.actions[:batch_size]
        self.states = self.states[:batch_size]
        self.episode_tensors = self.episode_tensors[:batch_size]
        self.logprobs = self.logprobs[:batch_size]
        self.rewards = self.rewards[:batch_size]
        self.state_values = self.state_values[:batch_size]
        self.is_terminals = self.is_terminals[:batch_size]
        self.calc_time = self.calc_time[:batch_size]
        self.makespan = self.makespan[:batch_size]
        self.num_collision_agents = self.num_collision_agents[:batch_size]

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.episode_tensors[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.calc_time[:]
        del self.makespan[:]
        del self.num_collision_agents[:]        
        
def concate_buffers(buffers: List[RolloutBuffer]):
    buffer = buffers[0]
    for i in range(1, len(buffers)):
        buffer += buffers[i]
    return buffer


def calc_mc_returns(buffer:RolloutBuffer, gamma, device, normalize=True):
    # Monte Carlo estimate of returns
    returns = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        returns.append(discounted_reward)
    
    returns.reverse()
    
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    if normalize:
        # Normalizing the rewards
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        returns = returns / (returns.std() + 1e-7)
    return returns


def translate2tensor(buffer:RolloutBuffer, gamma, device, normalized_return=True):
    B = len(buffer.episode_tensors) # batch size
    T = N =  len(buffer.states) // B  # time, num of agents 
    
    returns = calc_mc_returns(buffer, gamma, device, normalized_return).view(B, T).detach().to(device)

    rewards =torch.stack(buffer.rewards, dim=0).view(B, T).detach().to(device)

    old_actions = torch.stack(buffer.actions, dim=0).view(B, T).detach().to(device)
    
    old_logprobs = torch.stack(buffer.logprobs, dim=0
        ).view(B,T).detach().to(device) if len(buffer.logprobs) >0 else None
    old_state_values = torch.stack(buffer.state_values, dim=0
        ).view(B,T).detach().to(device) if len(buffer.logprobs) >0 else None

    # calculate advantages
    advantages = returns.detach() - old_state_values.detach() if old_state_values is not None else None
    
    episode_tensors = EpisodeTensors(buffer.episode_tensors).detach().to(device)
    state_tensors = [s.detach().to(device) for s in buffer.states]
    
    return (returns, rewards, old_actions, old_logprobs, old_state_values, 
            advantages, episode_tensors, state_tensors)