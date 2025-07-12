import torch
import torch.nn as nn
import torch.nn.functional as F




class PrototypeInfoNCE(nn.Module):
    def __init__(self,temperature=0.7):
        
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features,prototypes, labels):
        
        prototypes = prototypes.to(features.device)
        
        
        features_norm = F.normalize(features, dim=1)  # [batch_size, feature_dim]
        prototypes_norm = F.normalize(prototypes, dim=1)  # [num_classes, feature_dim]
        
        
        sim_matrix = torch.mm(features_norm, prototypes_norm.t()) / self.temperature
        
        pos_sim = torch.sum(labels * torch.exp(sim_matrix), dim=1)
      
        all_sim = torch.sum(torch.exp(sim_matrix), dim=1)
        
        per_sample_loss = -torch.log(pos_sim / all_sim)
        
        return per_sample_loss.mean()