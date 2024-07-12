import torch
from torch import nn
from torchvision import models

# implements Map and optimal path info
class Perception(nn.Module):
    # all info to [b, num_agents + 1, d_model].
    def __init__(self, out_features=256) -> None:
        super(Perception, self).__init__()
        self.vision = models.resnet18(num_classes=out_features)

        # 2 : 2 : 1
        loc_size = int(out_features * 2 / 5)
        self.loc_emb = nn.Linear(2, loc_size)
        harm_size = out_features - loc_size *2 
        self.harm_embedding = nn.Embedding(2, harm_size)

    def forward(self, features_map:torch.Tensor, starts, goals, harmful:torch.Tensor):
        # features_map: [b, 3, wid, height]
        # starts, goals: [b, n, 2]. 
        # harmful: [b, n, 1]
        
        map_emb = self.vision(features_map.type(torch.float))

        starts_emb = self.loc_emb(starts)
        goals_emb = self.loc_emb(goals)
        harm_emb = self.harm_embedding(harmful.long().squeeze(-1))

        x = torch.cat([starts_emb, goals_emb,harm_emb], dim=-1)
        x= torch.cat([map_emb.unsqueeze(1), x], dim=1)

        return x[:,1:,...]
    
class PerceptionPureHarmful(nn.Module):
    def __init__(self, out_features=256) -> None:
        super(PerceptionPureHarmful, self).__init__()
        # self.vision = models.resnet18()
        # self.fc = nn.Linear(1000, out_features-1)
        self.embedding = nn.Embedding(2, out_features)
        self.out_features = out_features

    def forward(self, harmful:torch.Tensor):
        # features_map: [b, n, 3, wid, height]
        # harmful: [b, n, 1]
        x = self.embedding(harmful.long().squeeze(-1))
        return x

class PerceptionRedudant(nn.Module):
    # all info to [b, num_agents + 1, d_model].
    def __init__(self, out_features=256) -> None:
        super(Perception, self).__init__()
        vision_sz = int(out_features/4*3)
        harm_size = out_features - vision_sz
        self.vision = models.resnet18(num_classes=out_features)

        # 3 : 1
        self.harm_embedding = nn.Embedding(2, harm_size)

    def forward(self, features_map:torch.Tensor, harmful:torch.Tensor):
        # features_map: [b, n, 3, wid, height]
        # starts, goals: [b, n, 2]. 
        # harmful: [b, n, 1]
        assert len(features_map.shape) == 4, "features map must be [b,n,3,wid,height]"

        B, N = [*features_map.shape[:2]]
        features_map.view(-1, *features_map.shape[2:])
        map_emb = self.vision(features_map.type(torch.float)).view(B,N,-1)
        harm_emb = self.harm_embedding(harmful.long().squeeze(-1))
        x = torch.cat([map_emb, harm_emb], dim=-1)

        return x
    