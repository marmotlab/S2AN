import torch
from torch import nn
from typing import List

from nets.inti_embeding import Perception, PerceptionPureHarmful, PerceptionRedudant
from rollout_buffer import EpisodeTensors, StateTensor


class SimpleLinear(nn.Module):

    def __init__(self,
                 embedding_dim,
                 init_embed_model_str="Perception",
                 n_encode_layers=2,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 tanh_clip=50,
                 device=torch.device('cpu')):
        super(SimpleLinear, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.tanh_clip = tanh_clip
        
        self.init_embed_model_str = init_embed_model_str
        init_embed_model = {
            "Perception" : Perception,
            "PerceptionRedudant": PerceptionRedudant,
            "PerceptionPureHarmful": PerceptionPureHarmful
        }.get(init_embed_model_str, None)
        assert init_embed_model is not None,  init_embed_model_str + " is not implemented."
        
        self.init_embed = init_embed_model(embedding_dim)

        self.l1 = nn.Linear(embedding_dim, embedding_dim)
        self.l2 = nn.Linear(embedding_dim, 1)

        self.critic = nn.Linear(embedding_dim, 1)

        self.device = device
        self.fixed = None
    
    def reset(self):
        self.fixed = None
    
    def forward(self, epts:EpisodeTensors,  
                states:List[StateTensor], inplace=True):
        x = self._init_embed(epts)
        x = torch.relu(self.l1(x))
        action_probs = torch.softmax(self.l2(x).squeeze(), dim=-1)
        state_vals = self.critic(x).squeeze()
        B = epts.shape[0]
        T = N = 100
        return action_probs.reshape(B, 1, N).expand(-1,T,-1), state_vals
        
    
    def _init_embed(self, epts: EpisodeTensors):
        if isinstance(self.init_embed, Perception):
            x = self.init_embed(epts.features_map, epts.starts, epts.goals, 
                                epts.harmful)
        elif isinstance(self.init_embed, PerceptionRedudant):
            x = self.init_embed(epts.features_map, epts.harmful)
        elif isinstance(self.init_embed, PerceptionPureHarmful):
            x= self.init_embed(epts.harmful)
        else:
            x = None
        return x