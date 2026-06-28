
from torch import nn
from core.model.backbone.sage import SAGEEncoder


class Encoder(nn.Module):
    def __init__(self, 
                 alert_embedding_dim: int,
                 graph_hidden_dim: int, 
                 graph_out_dim: int,
                 num_layers=2,
                 aggregator='mean',
                 feat_drop=0.3):
        super(Encoder, self).__init__()

        self.graph_encoder = SAGEEncoder(
            in_dim=alert_embedding_dim,
            out_dim=graph_out_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=num_layers,
            aggregator_type=aggregator,
            feat_drop=feat_drop
        )


    def forward(self, g, x):
        f, e = self.graph_encoder(g, x)
        return f, e