from torch_geometric.nn import GCNConv, Sequential, Linear
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, gcn_layers=2, transformer_layers=2, dropout=0.0, activation='relu', nhead=4):
        super(TransGCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.transformers = torch.nn.ModuleList()

        # 第一个 GCN 层
        self.convs.append(GCNConv(num_features, hidden_channels))

        # 交替添加 GCN 层和 Transformer 层
        for _ in range(1, gcn_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            transformer_layer = TransformerEncoderLayer(d_model=hidden_channels, nhead=nhead, dropout=dropout)
            self.transformers.append(TransformerEncoder(transformer_layer, num_layers=transformer_layers))

        self.dropout = dropout
        if activation not in ['relu', 'leaky_relu', 'sigmoid', 'tanh']:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'leaky_relu', 'sigmoid', 'tanh'.")
        self.activation = getattr(F, activation if activation != 'leaky_relu' else 'leaky_relu_')
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # 残差连接需要保持 x 的原始副本
        for conv, trans in zip(self.convs, self.transformers):
            identity = x  # 保存输入作为残差
            x = conv(x, edge_index)
            x = self.activation(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            # 应用 Transformer 并加入残差
            x = trans(x.unsqueeze(0)).squeeze(0) + x  # 加入残差
            x = self.activation(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, gcn_layers=2, dropout=0.0, activation='relu'):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(1, gcn_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        if activation not in ['relu', 'leaky_relu', 'sigmoid', 'tanh']:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'leaky_relu', 'sigmoid', 'tanh'.")
        self.activation = getattr(F, activation if activation != 'leaky_relu' else 'leaky_relu_')
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, **kwargs):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=4, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=True, dropout=0.6)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)


from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, **kwargs):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)

from torch_geometric.nn import GINConv, MLP

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels,  **kwargs):
        super(GIN, self).__init__()
        self.conv1 = GINConv(MLP([num_features, hidden_channels, hidden_channels]), train_eps=True)
        self.conv2 = GINConv(MLP([hidden_channels, hidden_channels, hidden_channels]), train_eps=True)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)


from torch_geometric.nn import MessagePassing
import torch_geometric.utils as utils

class GeneralGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GeneralGNN, self).__init__()
        self.conv1 = MessagePassing(aggr='add')  # 使用加法聚合
        self.fc1 = torch.nn.Linear(num_features, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.fc1(x)
        edge_index, _ = utils.remove_self_loops(edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)
#
# from torch_geometric.nn import DiffConv
#
# class DCNN(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels):
#         super(DCNN, self).__init__()
#         self.conv1 = DiffConv(num_features, hidden_channels)
#         self.conv2 = DiffConv(hidden_channels, hidden_channels)
#         self.out = torch.nn.Linear(hidden_channels, 1)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         return x
#
#     def predict_link(self, x, edge_index):
#         row, col = edge_index
#         edge_features = x[row] * x[col]
#         return self.out(edge_features)
#
from torch_geometric.nn import ChebConv  # Chebyshev polynomials
import torch.nn as nn

class GCRN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCRN, self).__init__()
        self.conv = ChebConv(num_features, hidden_channels, K=2)
        self.rnn = nn.GRUCell(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)
        self.hidden = self.init_hidden()

    def forward(self, x, edge_index, hidden):
        x = F.relu(self.conv(x, edge_index))
        hidden = self.rnn(x, hidden)
        return hidden

    def init_hidden(self, batch_size):
        # 初始化隐藏状态为全零
        # 这里的 '2' 代表双向RNN，如果是单向的就用 '1'
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)


from torch_geometric.nn import EdgeConv, DynamicEdgeConv

class EdgeCNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(EdgeCNN, self).__init__()
        self.conv = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * num_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), k=6)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv(x)
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)


from torch_geometric.nn import TopKPooling

class TopoNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(TopoNN, self).__init__()
        self.conv = GCNConv(num_features, hidden_channels)
        self.pool = TopKPooling(hidden_channels, ratio=0.8)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)

from torch_geometric.nn import ChebConv

class ChebNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, K=2):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(num_features, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)

from torch_geometric.nn import EdgeConv

class DGCNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(DGCNN, self).__init__()
        self.conv1 = EdgeConv(MLP([2 * num_features, hidden_channels, hidden_channels]), aggr='max')
        self.conv2 = EdgeConv(MLP([2 * hidden_channels, hidden_channels, hidden_channels]), aggr='max')
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def predict_link(self, x, edge_index):
        row, col = edge_index
        edge_features = x[row] * x[col]
        return self.out(edge_features)
