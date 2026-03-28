import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class GeometryFeatureMerger(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 4096):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, geometry_tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if geometry_tokens.dim() == 2:
            geometry_tokens = geometry_tokens.unsqueeze(0)

        geometry_tokens = self.proj(self.norm(geometry_tokens))
        geometry_tokens = geometry_tokens.transpose(1, 2)
        geometry_tokens = F.adaptive_avg_pool1d(geometry_tokens, target_tokens)
        geometry_tokens = geometry_tokens.transpose(1, 2)
        return geometry_tokens


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
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

    def forward(self, image_tokens: torch.Tensor, geometry_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(image_tokens)
        key = self.key_norm(geometry_tokens)
        value = key
        attn_output, _ = self.attn(query, key, value)
        fused = image_tokens + attn_output
        fused = fused + self.mlp(self.out_norm(fused))
        return fused


class FeatureFusionModule(nn.Module):
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method

        if self.fusion_method == "gated":
            self.image_norm = nn.LayerNorm(config.hidden_size)
            self.geometry_norm = nn.LayerNorm(config.hidden_size)
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Sigmoid(),
            )
        elif self.fusion_method == "concat":
            self.image_norm = nn.LayerNorm(config.hidden_size)
            self.geometry_norm = nn.LayerNorm(config.hidden_size)
            self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        elif self.fusion_method == "cross_attention":
            self.blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        dropout=config.dropout,
                    )
                    for _ in range(config.num_layers)
                ]
            )

    def forward(self, image_tokens: torch.Tensor, geometry_tokens: torch.Tensor) -> torch.Tensor:
        if image_tokens.dim() == 2:
            image_tokens = image_tokens.unsqueeze(0)
        if geometry_tokens.dim() == 2:
            geometry_tokens = geometry_tokens.unsqueeze(0)

        if self.fusion_method == "add":
            fused = image_tokens + geometry_tokens
        elif self.fusion_method == "concat":
            fused = self.proj(
                torch.cat(
                    [self.image_norm(image_tokens), self.geometry_norm(geometry_tokens)],
                    dim=-1,
                )
            )
        elif self.fusion_method == "gated":
            norm_image = self.image_norm(image_tokens)
            norm_geometry = self.geometry_norm(geometry_tokens)
            gate = self.gate(torch.cat([norm_image, norm_geometry], dim=-1))
            fused = gate * norm_image + (1.0 - gate) * norm_geometry
        elif self.fusion_method == "cross_attention":
            fused = image_tokens
            for block in self.blocks:
                fused = block(fused, geometry_tokens)
        elif self.fusion_method == "only_3d":
            fused = geometry_tokens
        elif self.fusion_method == "zero":
            fused = image_tokens
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused
