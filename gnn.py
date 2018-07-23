import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Graph_conv_block(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True):
        super(Graph_conv_block, self).__init__()

        self.weight = nn.Linear(input_dim, output_dim)
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = None

    def forward(self, x, A):
        x_next = torch.matmul(A, x) # (b, N, input_dim)
        x_next = self.weight(x_next) # (b, N, output_dim)

        if self.bn is not None:
            x_next = torch.transpose(x_next, 1, 2) # (b, output_dim, N)
            x_next = x_next.contiguous()
            x_next = self.bn(x_next)
            x_next = torch.transpose(x_next, 1, 2) # (b, N, output)

        return x_next

class Adjacency_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, ratio=[2,2,1,1]):

        super(Adjacency_layer, self).__init__()

        module_list = []

        for i in range(len(ratio)):
            if i == 0:
                module_list.append(nn.Conv2d(input_dim, hidden_dim*ratio[i], 1, 1))
            else:
                module_list.append(nn.Conv2d(hidden_dim*ratio[i-1], hidden_dim*ratio[i], 1, 1))

            module_list.append(nn.BatchNorm2d(hidden_dim*ratio[i]))
            module_list.append(nn.LeakyReLU())

        module_list.append(nn.Conv2d(hidden_dim*ratio[-1], 1, 1, 1))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        X_i = x.unsqueeze(2) # (b, N , 1, input_dim)
        X_j = torch.transpose(X_i, 1, 2) # (b, 1, N, input_dim)

        phi = torch.abs(X_i - X_j) # (b, N, N, input_dim)

        phi = torch.transpose(phi, 1, 3) # (b, input_dim, N, N)

        A = phi

        for l in self.module_list:
            A = l(A)
        # (b, 1, N, N)

        A = torch.transpose(A, 1, 3) # (b, N, N, 1)

        A = F.softmax(A, 2) # normalize

        return A.squeeze(3) # (b, N, N)

class GNN_module(nn.Module):
    def __init__(self, nway, input_dim, hidden_dim, num_layers, feature_type='dense'):
        super(GNN_module, self).__init__()

        self.feature_type = feature_type

        adjacency_list = []
        graph_conv_list = []

        # ratio = [2, 2, 1, 1]
        ratio = [2, 1]

        if self.feature_type == 'dense':
            for i in range(num_layers):
                adjacency_list.append(Adjacency_layer(
                    input_dim=input_dim+hidden_dim//2*i, 
                    hidden_dim=hidden_dim, 
                    ratio=ratio))

                graph_conv_list.append(Graph_conv_block(
                    input_dim=input_dim+hidden_dim//2*i, 
                    output_dim=hidden_dim//2))

            # last layer
            last_adjacency = Adjacency_layer(
                        input_dim=input_dim+hidden_dim//2*num_layers, 
                        hidden_dim=hidden_dim, 
                        ratio=ratio)

            last_conv = Graph_conv_block(
                    input_dim=input_dim+hidden_dim//2*num_layers, 
                    output_dim=nway, 
                    use_bn=False)

        elif self.feature_type == 'forward':
            for i in range(num_layers):
                adjacency_list.append(Adjacency_layer(
                    input_dim=input_dim if i == 0 else hidden_dim, 
                    hidden_dim=hidden_dim, 
                    ratio=ratio))

                graph_conv_list.append(Graph_conv_block(
                    input_dim=hidden_dim, 
                    output_dim=hidden_dim))

            # last layer
            last_adjacency = Adjacency_layer(
                        input_dim=hidden_dim, 
                        hidden_dim=hidden_dim, 
                        ratio=ratio)

            last_conv = Graph_conv_block(
                    input_dim=hidden_dim, 
                    output_dim=nway,
                    use_bn=False)

        else:
            raise NotImplementedError

        self.adjacency_list = nn.ModuleList(adjacency_list)
        self.graph_conv_list = nn.ModuleList(graph_conv_list)
        self.last_adjacency = last_adjacency
        self.last_conv = last_conv


    def forward(self, x):
        for i, _ in enumerate(self.adjacency_list):
            adjacency_layer = self.adjacency_list[i]
            conv_block = self.graph_conv_list[i]

            A = adjacency_layer(x)

            x_next = conv_block(x, A)

            x_next = F.leaky_relu(x_next, 0.1)

            if self.feature_type == 'dense':
                x = torch.cat([x, x_next], dim=2)
            elif self.feature_type == 'forward':
                x = x_next
            else:
                raise NotImplementedError
        
        A = self.last_adjacency(x)
        out = self.last_conv(x, A)   

        return out[:, 0, :]
