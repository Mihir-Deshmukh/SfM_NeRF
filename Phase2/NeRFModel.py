import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_dir_L=4, hidden_size=256):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        # Define the layers according to the provided filter size and input dimension
        
        self.fc_input_dim = 3 + 3 * 2 * embed_pos_L
        # self.fc_input_dim = 3
        self.fc_feat_input_dim = 3 + 3 * 2 * embed_dir_L
        # self.fc_feat_input_dim = 3

        # Define the MLP
        self.layers = nn.ModuleList()
        for i in range(8):
            in_features = self.fc_input_dim if i == 0 else hidden_size
            if i in [4]:
                in_features += self.fc_input_dim
                
            if i in [7]:
                out_features = hidden_size + 1
            else:
                out_features = hidden_size
            self.layers.append(nn.Linear(in_features, out_features))
        
        
        self.feat_layer = nn.Linear(hidden_size + self.fc_feat_input_dim, hidden_size//2)
        # Output layer
        self.rgb_layer = nn.Linear(hidden_size//2, 3)
        
        # Store the positional encoding length
        self.embed_pos_L = embed_pos_L
        self.embed_dir_L = embed_dir_L

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

    def forward(self, pos, dir):
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
        
        sigma, x = x[..., -1], x[..., :-1]
        x = torch.cat([x, self.position_encoding(dir, self.embed_dir_L)], -1)
        x = F.relu(self.feat_layer(x))
        x = self.rgb_layer(x)

        return F.sigmoid(x), sigma
    
    # def forward(self, pos, dir):
    #     #############################
    #     # network structure
    #     #############################
    #     # print(f"pos: {pos.shape} ")
    #     x = pos
    #     # print(f"x: {x.shape}")
    #     for i, layer in enumerate(self.layers):
    #         if i in [4] and i > 0:
    #             x = torch.cat([x, pos], -1)
    #         x = F.relu(layer(x))
        
    #     sigma, x = x[..., -1], x[..., :-1]
    #     x = torch.cat([x, dir], -1)
    #     x = F.relu(self.feat_layer(x))
    #     x = self.rgb_layer(x)

    #     return F.sigmoid(x), sigma
    

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
    