from dataclasses import dataclass

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT


@dataclass
class FeatureFusionConfig:
    fusion_method: str = "gated"
    hidden_size: int = 4096
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class VGGTGeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggt = VGGT(
            img_size=224,
            patch_size=14,
            embed_dim=1024,
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )
        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False

    @property
    def patch_size(self):
        return 14

    @property
    def feature_dim(self):
        return 2048

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)

        param = next(self.vggt.parameters())
        images = images.to(device=param.device, dtype=param.dtype)

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
            features = aggregated_tokens_list[-2][0, :, patch_start_idx:]
        return features


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_3d)
        value = self.norm1_value(features_3d)
        attn_output, _ = self.cross_attention(query, key, value)
        x = features_2d + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size

        if self.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif self.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        hidden_size=self.hidden_size,
                        num_heads=self.config.num_heads,
                        dropout=self.config.dropout,
                    )
                    for _ in range(self.config.num_layers)
                ]
            )
        elif self.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid(),
            )
        elif self.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.0))
            self.weight_3d = nn.Parameter(torch.tensor(1.0))

    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        if features_2d.dim() == 4:
            b, h, w, _ = features_2d.shape
            features_2d_seq = features_2d.reshape(b, h * w, -1)
        else:
            b = features_2d.shape[0]
            features_2d_seq = features_2d

        if features_3d.dim() == 4:
            _, h3, w3, _ = features_3d.shape
            features_3d_seq = features_3d.reshape(b, h3 * w3, -1)
        else:
            features_3d_seq = features_3d

        if self.fusion_method == "add":
            fusion_feature = features_2d_seq + features_3d_seq
        elif self.fusion_method == "concat":
            fusion_feature = self.projection(
                torch.cat([self.norm1(features_2d_seq), self.norm2(features_3d_seq)], dim=-1)
            )
        elif self.fusion_method == "cross_attention":
            x = features_2d_seq
            for block in self.cross_attn_blocks:
                x = block(x, features_3d_seq)
            fusion_feature = x
        elif self.fusion_method == "gated":
            norm2d = self.norm1(features_2d_seq)
            norm3d = self.norm2(features_3d_seq)
            gate = self.gate_projection(torch.cat([norm2d, norm3d], dim=-1))
            fusion_feature = gate * norm2d + (1 - gate) * norm3d
        elif self.fusion_method == "weighted":
            weight_sum = self.weight_2d + self.weight_3d + 1e-6
            w2d = self.weight_2d / weight_sum
            w3d = self.weight_3d / weight_sum
            fusion_feature = w2d * features_2d_seq + w3d * features_3d_seq
        elif self.fusion_method == "only_3d":
            fusion_feature = features_3d_seq
        elif self.fusion_method == "zero":
            fusion_feature = features_2d_seq
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        if features_2d.dim() == 4:
            return fusion_feature.reshape(b, h, w, -1)
        return fusion_feature


class GeometryFeatureMerger(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        merger_type: str = "mlp",
    ):
        super().__init__()
        self.merger_type = merger_type
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.merge_size = spatial_merge_size
        self.input_dim = context_dim * (spatial_merge_size ** 2)

        if merger_type == "mlp":
            self.norm = nn.LayerNorm(context_dim)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type != "avg":
            raise ValueError(f"Unknown merger type: {merger_type}")

    def forward(self, x: torch.Tensor, target_hw=None):
        n_image, h_patch, w_patch, dim = x.shape

        merge_size = self.merge_size
        if target_hw is not None:
            target_h, target_w = target_hw
            if target_h > 0 and target_w > 0:
                merge_h = max(1, h_patch // target_h)
                merge_w = max(1, w_patch // target_w)
                if merge_h == merge_w and merge_h > 0:
                    merge_size = merge_h

        h_valid = (h_patch // merge_size) * merge_size
        w_valid = (w_patch // merge_size) * merge_size
        x = x[:, :h_valid, :w_valid, :]
        x = x.reshape(n_image, h_valid // merge_size, merge_size, w_valid // merge_size, merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        if self.merger_type == "mlp":
            if merge_size != self.merge_size:
                raise ValueError(
                    f"GeometryFeatureMerger merge_size mismatch: expected {self.merge_size}, got {merge_size}"
                )
            x_flat = self.norm(x).view(-1, self.input_dim)
            x_flat = self.mlp(x_flat)
        else:
            x_flat = x.mean(dim=(3, 4))
            x_flat = x_flat.view(-1, dim)
            x_flat = self.mlp(x_flat)

        x = x_flat.reshape(n_image, h_valid // merge_size, w_valid // merge_size, -1)
        return x
