import math
from torch.nn import functional as F
import torch
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphConvolution(nn.Module):
    """
    simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # 2560
        self.out_features = out_features  # 8
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):  # out==>x(32,17,2560);adj(64,17,17)
        # x(32,17,2560) * weight(2560,8) - bias(1,1,8)==>output(32,17,8)
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))  # adj(32,17,17)*output(32,17,8)==>(32,17,8) (这两行代码完成公式(14)的操作)
        return output  # (64,17,8)


class makeAdj(nn.Module):

    def __int__(self):
        super(makeAdj, self).__init__()
        # self.global_adj = nn.Parameter(torch.FloatTensor(64, 64), requires_grad=True)
    def self_similarity(self, x):
        # print("正在通过LGG的self_similarity代码")
        # x: b, node, feature; x(32,17,2506);x_(32,2506,17)
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s  # s(32,17,17)

    def get_adj(self, x, self_loop=True):
        # print("正在通过LGG的get_adj代码")
        # x: b, node, feature ==>(32,17,2506)
        adj = self.self_similarity(x)  # 输出adj(b, n, n)即(64,17,17)表示为节点之间的自相似性矩阵
        num_nodes = adj.shape[-1]
        global_adj = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes)).to(device)
        torch.nn.init.xavier_uniform_(global_adj, gain=1.414)
        # 此步adj中就引入了全局信息,通过 self.global_adj + self.global_adj.transpose(1, 0) 的操作,将其与自身的转置相加,
        # 相当于对全局权重矩阵进行对称化处理,以便与 adj 进行元素级别的乘法运算
        adj = F.relu((adj + torch.eye(num_nodes).to(device)) * (
                global_adj + global_adj.transpose(1, 0)))  # global_adj(17,17);adj(64,17,17)
        # if self_loop:
        #     adj = adj + torch.eye(num_nodes).to(device)  # torch.eye生成单位阵 （到此完成了公式12的操作）
        rowsum = torch.sum(adj, dim=-1)  # 求行和, adj(64,17,17)==>rowsum(64,17)
        mask = torch.zeros_like(rowsum)  # 产生全为0的矩阵,大小等于rowsum(64,17)
        mask[rowsum == 0] = 1  # 遍历 rowsum 列表中的每个元素，如果元素的值为0，则将 mask 对应位置的元素赋值为1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)  # 计算每个元素的倒数的平方根,用于归一化邻接矩阵  d_inv_sqrt(64,17)
        # d_mat_inv_sqrt(64,17,17)  torch.diag_embed()函数作用是将指定值变成对角矩阵==>具体来说是将d_inv_sqrt的每行元素变成生成一个对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        # 对邻接矩阵adj进行对称归一化 (64,17,17)*(64,17,17) * (64,17,17)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj  # (64,17,17)

    def forward(self, x):
        return self.get_adj(x)

class GCN(nn.Module):
    """图卷积模块"""

    def __init__(self, in_features, attn_features):
        super(GCN, self).__init__()

        self.in_features = in_features
        self.attn_features = attn_features

        self.gconv1 = GraphConvolution(in_features, attn_features)

        self.adj = makeAdj()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """

        adjlist = self.adj(x)

        x = F.relu(self.gconv1(x, adjlist))

        return x  # 返回缩放后的特征