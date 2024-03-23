import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class MlpGeLUPostCrossAttProjector(nn.Module):
    def __init__(self, mlp_gelu_cross_att_match, config):
        super(MlpGeLUPostCrossAttProjector, self).__init__()

        mlp_depth = int(mlp_gelu_cross_att_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))

        self.projector = nn.Sequential(*modules)
        query_num = int(mlp_gelu_cross_att_match.group(3))
        self.query = nn.Parameter(torch.zeros(1, query_num, config.hidden_size))
        self.query.data.normal_(mean=0.0, std=0.02)

        att_layer_num = int(mlp_gelu_cross_att_match.group(2))
        decoder_layer = nn.TransformerDecoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=0,
            activation="gelu",
            batch_first=True,
        )
        self.cross_att_layers = nn.TransformerDecoder(
            decoder_layer, att_layer_num, norm=None
        )

    def forward(self, vision_embedding):
        projected_embedding = self.projector(vision_embedding)
        batch_size = vision_embedding.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        output = self.cross_att_layers(query, projected_embedding)
        return output

class MlpGeLUPreCrossAttProjector(nn.Module):
    def __init__(self, mlp_gelu_cross_att_match, config):
        super(MlpGeLUPreCrossAttProjector, self).__init__()

        mlp_depth = int(mlp_gelu_cross_att_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.projector = nn.Sequential(*modules)

        query_num = int(mlp_gelu_cross_att_match.group(3))
        self.query = nn.Parameter(torch.zeros(1, query_num, config.mm_hidden_size))
        self.query.data.normal_(mean=0.0, std=0.02)
        att_layer_num = int(mlp_gelu_cross_att_match.group(2))
        decoder_layer = nn.TransformerDecoderLayer(
            config.mm_hidden_size,
            8,
            dim_feedforward=config.mm_hidden_size * 4,
            dropout=0,
            activation="gelu",
            batch_first=True,
        )
        self.cross_att_layers = nn.TransformerDecoder(
            decoder_layer, att_layer_num, norm=None
        )

    def forward(self, vision_embedding):
        batch_size = vision_embedding.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        output = self.cross_att_layers(query, vision_embedding)
        output = self.projector(output)
        return output


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_pre_cross_att_match = re.match(
        r"^mlp(\d+)x_gelu_pre_(\d+)att_(\d+)q$", projector_type
    )
    if mlp_gelu_pre_cross_att_match:
        return MlpGeLUPreCrossAttProjector(mlp_gelu_pre_cross_att_match, config)

    mlp_gelu_post_cross_att_match = re.match(
        r"^mlp(\d+)x_gelu_post_(\d+)att_(\d+)q$", projector_type
    )
    if mlp_gelu_post_cross_att_match:
        return MlpGeLUPostCrossAttProjector(mlp_gelu_post_cross_att_match, config)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
