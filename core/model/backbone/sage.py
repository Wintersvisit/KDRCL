import torch.nn as nn
from torch.functional import F
import dgl.nn.pytorch as dglnn

class SAGEEncoder(nn.Module):
    def __init__(self, 
                 in_dim,
                 hidden_dim,                # 隐藏层维度，即中间过程特征的宽度
                 out_dim,
                 feat_drop=0.3,
                 aggregator_type='mean',    # GraphSAGE在收集邻居信息时，把所有邻居的特征向量取平均值
                 num_layers=2):             # 图卷积的层数，代表信息传递的“跳数”
        super(SAGEEncoder, self).__init__()

        self.num_layers=num_layers
        self.sage_layers = nn.ModuleList()
        self.activation = F.relu

        if num_layers == 1:
            self.sage_layers.append(
                dglnn.SAGEConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    feat_drop=feat_drop,
                    aggregator_type=aggregator_type,
                    activation=None,
                    bias=False
                )
            )
        else:
            self.sage_layers.append(            # 输入层 (in_dim -> hidden_dim)
                dglnn.SAGEConv(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    feat_drop=feat_drop,
                    activation=self.activation,
                    aggregator_type=aggregator_type,
                    bias=False
                )
            )

            for l in range(0, num_layers-2):    # 中间隐藏层 (hidden_dim -> hidden_dim)
                self.sage_layers.append(
                    dglnn.SAGEConv(
                        in_feats=hidden_dim,
                        out_feats=hidden_dim,
                        feat_drop=feat_drop,
                        aggregator_type=aggregator_type,
                        activation=self.activation,
                        bias=False
                    )
                )
            self.sage_layers.append(            # 输出层 (hidden_dim -> out_dim)
                dglnn.SAGEConv(
                    in_feats=hidden_dim,
                    out_feats=out_dim,
                    feat_drop=feat_drop,
                    aggregator_type=aggregator_type,
                    activation=None,
                    bias=False
                )
            )
        
        self.pool=dglnn.MaxPooling()            # 全局最大池化，聚合图中所有节点的特征

    def forward(self, g, x):                    # 前向传播
        edge_weight = g.edata.get('weight', None)

        for l in range(self.num_layers - 1):
            x = self.sage_layers[l](g, x, edge_weight=edge_weight)
        e = self.sage_layers[-1](g, x, edge_weight=edge_weight)
        f = self.pool(g, e)
        return f, e
