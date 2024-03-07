import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# class NeRFmodel(nn.Module):
#     def __init__(self, embed_pos_L = 10, embed_direction_L = 4):
#         super(NeRFmodel, self).__init__()
#         #############################
#         # network initialization
#         #############################
        
#         self.embedding_dim_o = embed_pos_L
#         self.embedding_dim_d = embed_direction_L

#     def position_encoding(self, x, L):
#         #############################
#         # Implement position encoding here
#         #############################
#         out = [x]
#         for i in range(L):
#             out.append(torch.sin(2**i * x))
#             out.append(torch.cos(2**i * x))

#         y = torch.cat(out, dim=-1)
#         return y

#     def forward(self, pos, direction):
#         #############################
#         # network structure
#         #############################
#         pos_embed_origins = self.position_encoding(pos, self.embedding_dim_o)
#         pos_embed_directions = self.position_encoding(direction, self.embedding_dim_d)
        
#         x = pos_embed_origins
#         for i, layer in enumerate(self.position_layers):
#             x = F.relu(layer(x))
#             if i in self.skips:
#                 x = torch.cat([pos_embed_origins, x], -1)

#         sigma = self.sigma_layer(x)
#         x = torch.cat([x, pos_embed_directions], -1)
        
#         rgb = self.direction_layers(x)

#         outputs = torch.cat([rgb, sigma], -1)

#         return out
    
    

class TinyNeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, hidden_size=256):
        super(TinyNeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        # Define the layers according to the provided filter size and input dimension
        self.fc_input_dim = 3 + 3 * 2 * embed_pos_L
        
        # Define the MLP
        self.layers = nn.ModuleList()
        for i in range(8):
            in_features = self.fc_input_dim if i == 0 else hidden_size
            if i in [4]:
                in_features += self.fc_input_dim
            self.layers.append(nn.Linear(in_features, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 4)
        
        # Store the positional encoding length
        self.embed_pos_L = embed_pos_L

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################
        out = [x]
        for i in range(L):
            out.append(torch.sin(2**i * np.pi * x))
            out.append(torch.cos(2**i * np.pi * x))

        y = torch.cat(out, dim=-1)
        return y

    def forward(self, pos):
        #############################
        # network structure
        #############################
        # print(f"pos: {pos.shape} ")
        x = self.position_encoding(pos, self.embed_pos_L)
        # print(f"x: {x.shape}")
        for i, layer in enumerate(self.layers):
            if i in [4] and i > 0:
                x = torch.cat([x, self.position_encoding(pos, self.embed_pos_L)], -1)
            x = F.relu(layer(x))
        
        x = self.output_layer(x)

        return F.sigmoid(x[..., :3]), F.relu(x[..., 3])
