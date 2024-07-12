import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple, List


from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel


from nets.inti_embeding import Perception, PerceptionPureHarmful, PerceptionRedudant
from rollout_buffer import StateTensor, EpisodeTensors

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    glimpse_query: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key : torch.Tensor
    logit_val : torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            glimpse_query=self.glimpse_query[:, key],  # dim 0 are the heads
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key], 
            logit_val=self.logit_val[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 init_embed_model_str="Perception",
                 n_encode_layers=2,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 tanh_clip=50,
                 device=torch.device('cpu')):
        super(AttentionModel, self).__init__()

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

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        self.atten_combine = nn.Linear(2,1, bias=False)
        # self.atten_combine2 = nn.Linear(12,1, bias=False)
        
        # For each node we compute (glimpse query, glimpse key, glimpse value) so 3 * embedding_dim
        # Wq, Wk, Wv, logit_k for final output.
        self.project_node_embeddings = nn.Linear(embedding_dim, 5 * embedding_dim, bias=False)

        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        # self.actor = nn.Linear(embedding_dim, 1, bias=False)
        self.Wo = nn.Linear(embedding_dim, embedding_dim)
        self.critic = nn.Linear(embedding_dim, 1)

        self.device = device
        self.fixed = None

    def reset(self):
        self.fixed = None

    def forward(self, epts:EpisodeTensors,  
                states:List[StateTensor], inplace=True):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        self.encode(epts)        
        num_agents = epts.num_agents
        batch_size =  epts.starts.shape[0]
        action_probs, state_vals = self.decode(
            epts, states, num_agents, batch_size, inplace=inplace)
       
        return action_probs, state_vals 

    def encode(self, epts:EpisodeTensors):
        # features_map: [b, 3, wid, height]
        # starts, goals: [b, n, 2]. 
        # harmful: [b, n, 1]
        
        # embeddings [b,n , embedding_size]
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(epts))
        else:
            embeddings, _ = self.embedder(self._init_embed(epts))
        # embeddings [b,n,z]
        
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_query_fixed, glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed, logit_val_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(5, dim=-1)
        
        num_steps = 1
        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_query_fixed, num_steps),
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed,
            logit_val_fixed
        )
        self.fixed = AttentionModelFixed(embeddings, *fixed_attention_node_data)

    def decode(self, episode_tensors:EpisodeTensors ,states:List[StateTensor], num_agents, batch_size, inplace=False):
        '''
        agents_planned: [b, t]. which agent is seleceted in batch in ti time.

        '''
        assert self.fixed is not None, "please encode before decode"
        # mask [b, t, n]
        mask = self.generate_mask(
            [s.agents_planned for s in states], batch_size, num_agents, self.device)
        time = len(states) // batch_size
        val_size = self.fixed.glimpse_query.shape[-1]  # val_size = feature_size//n_head
        action_probs = []
        state_vals = []
        
        B, T, N, H = batch_size, time, num_agents, self.n_heads

        # collision_atten = episode_tensors.target_matrix.view(B,1,N,N).expand(H, B,1,N,N)
        collision_atten = episode_tensors.target_matrix.view(B,1,N,N)
        
        query ,key, val, logit_key = self._get_fixed_data()
        for ti in range(time):
            if inplace:
                query = query.masked_fill_(mask[:,ti ].view(1,B,1,N,1)==0, 0)
                key = key.masked_fill_(mask[:,ti ].view(1,B,1,N,1)==0, 0)
                val = val.masked_fill_(mask[:,ti ].view(1,B,1,N,1)==0, 0)
                
                collision_atten = collision_atten.masked_fill_(mask[:,ti ].view(B,1,N,1)==0, 0)

            else:
                query = query.masked_fill(mask[:,ti ].view(1,B,1,N,1)==0, 0)
                key = key.masked_fill(mask[:,ti ].view(1,B,1,N,1)==0, 0)
                val = val.masked_fill(mask[:,ti ].view(1,B,1,N,1)==0, 0)
                collision_atten = collision_atten.masked_fill(mask[:,ti ].view(B,1,N,1)==0, 0)

            logit_key = logit_key.masked_fill(mask[:,ti ].view(B,1,N,1)==0, 0)
            
            d_k = query.size(-1)

            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            # reshape([1,B,1,1,N]) OR reshape([1,B,1,N,1])
            # scores = torch.stack([scores, collision_atten], dim=-1).view(H,B,1,N,N,2)
            # scores = self.atten_combine(scores).squeeze(-1)
            # scores =torch.relu( self.atten_combine(scores))
            # scores = self.atten_combine2(scores).squeeze(dim=-1)
            # scores = scores.masked_fill(mask[:,ti,].reshape([1,B,1,1,N])==0, -1e9)
            scores = scores.softmax(dim=-1)
            

            hidden = torch.matmul(scores, val)
            hidden = hidden.permute(1, 2, 3, 0, 4).contiguous().view(-1, 1, N, self.n_heads * val_size)
            # hidden = hidden.masked_fill(mask[:,ti].view(B,1,N,1)==0, 0)
            hidden = self.Wo(hidden)
            
            d_logit_k = logit_key.shape[-1]
            # hidden: final_q. single head attention in the final
            patten = torch.matmul(hidden, logit_key.transpose(-2, -1))/ math.sqrt(d_logit_k)
            patten = torch.stack([patten, collision_atten], dim=-1).view(B,1,N,N,2)
            patten = self.atten_combine(patten).squeeze(-1)
            
            action_prob = patten.sum(dim=-2).view(B,1,N)
            # action_prob = action_prob.softmax(dim=-1)
            # hidden = torch.matmul(hidden, self.fixed.logit_val)
            
            # action_prob = self.actor(hidden).view(B,1,N)
            action_prob = action_prob.masked_fill(mask[:,ti].view(B,1,N)==0, -1e9)
            action_prob = action_prob.softmax(dim=-1)
            
            state_val = torch.tanh(self.critic(hidden).view(B,1,N))*self.tanh_clip
            state_val = state_val.masked_fill(mask[:,ti].view(B,1,N)==0, 0)
            state_val = state_val.sum(dim=-1)/mask[:,ti].view(B,1,N).sum(dim=-1)
            action_probs.append(action_prob)
            state_vals.append(state_val)
        
        action_probs = torch.cat(action_probs, dim=1).view(B,T,N)
        state_vals = torch.cat(state_vals, dim=1).view(B,T)
        
        return action_probs, state_vals

    def generate_mask(self, agents_planned:List[torch.Tensor], 
                    batch_size, num_agents, device=torch.device('cpu')):
        '''agents_planned: index. have benn selected.
            len: batch*time.
        return: 
            mask. [batch_size, time, num_agents]
        '''
        batch_plus_time = len(agents_planned)
        mask = torch.ones([batch_plus_time, num_agents], dtype=torch.bool, device=device)
        # https://discuss.pytorch.org/t/fill-value-to-matrix-based-on-index/34698
        # mask[torch.arange(mask.shape[0]).unsqueeze(1), agents_planned] = 0
        for i, index in enumerate(agents_planned):
            mask[i].index_fill_(0, index.type(torch.int64), 0)  # dim, index, fill value

        return mask.reshape([batch_size, -1, num_agents])

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
    
    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
    
    def _get_fixed_data(self):
        return (
            self.fixed.glimpse_query,
            self.fixed.glimpse_key,
            self.fixed.glimpse_val,
            self.fixed.logit_key
        )

