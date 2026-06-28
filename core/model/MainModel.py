import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super(MainModel, self).__init__()
        self.config = config

        self.encoders = nn.ModuleDict()
        for modality in config.modalities:
            self.encoders[modality] = Encoder(
                alert_embedding_dim=config.alert_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop
            )

        rcl_fuse_dim = len(config.modalities) * config.graph_out
        self.graph_fuse_dim = len(config.modalities) * config.graph_out

        # 根因定位器
        self.locator = Voter(rcl_fuse_dim,
                             hiddens=config.linear_hidden,
                             out_dim=1)

        if config.feature_alignment:        # 如果开启特征对齐（对齐BERT的输出维度）
            self.graph_projection = torch.nn.Linear(
                self.graph_fuse_dim,
                config.explanation_dim
            ) if config.explanation_dim != self.graph_fuse_dim else None

    def forward(self, batch_graphs):
        fs, es = {}, {}
        for modality, encoder in self.encoders.items():   # 三种模态并行编码
            x_d = batch_graphs.ndata[modality]          # 取该模态的节点特征
            f_d, e_d = encoder(batch_graphs, x_d)
            fs[modality] = f_d            # 图级别特征 [图数, 32]（用于分类）
            es[modality] = e_d            # 节点级别特征 [图数, 32]（用于定位）

        e = torch.cat(list(es.values()), dim=1)         # 拼接三种模态节点特征
        root_logit = self.locator(e)            # 根因预测，[N, 96] → [N, 1]

        # 融合图级别特征（用于特征对齐）
        graph_embeddings = torch.cat(list(fs.values()), dim=1)

        if self.config.feature_alignment and self.graph_projection is not None:
            graph_embeddings = self.graph_projection(graph_embeddings)

        return graph_embeddings, e, root_logit, None
        """graph_embeddings: 融合后的图级别特征 [N, graph_fuse_dim],用于和 BERT 解释做特征对齐（对齐损失）
           e: 融合三种模态后的节点级别特征 [N, graph_fuse_dim]
           root_logit: 根因预测结果 [N, 1]"""