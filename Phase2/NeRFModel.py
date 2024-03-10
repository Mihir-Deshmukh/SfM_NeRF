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

class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):   
        super(NerfModel, self).__init__()
        
        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        # density estimation
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )
        # color estimation
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos) # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(d, self.embedding_dim_direction) # emb_d: [batch_size, embedding_dim_direction * 6]
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma
   
    

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
    
    
class VeryTinyNerfModel(torch.nn.Module):

    def __init__(self, filter_size=256, num_encoding_functions=10):
        super(VeryTinyNerfModel, self).__init__()
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu
    
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
  
    def forward(self, x):
        x = self.position_encoding(x, 10)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return torch.sigmoid(x[..., :3]), torch.relu(x[..., 3])

