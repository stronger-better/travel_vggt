import logging
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file

from vggt.models.vggt import VGGT


logger = logging.getLogger(__name__)

DEFAULT_VGGT_MODEL_REPO = "facebook/VGGT-1B"
DEFAULT_VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
LOCAL_VGGT_CANDIDATE_FILES = ("model.safetensors", "model.pt")


def _extract_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model_state_dict", "model", "module", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _strip_state_dict_prefix(state_dict, prefix):
    if not prefix:
        return state_dict

    if not any(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict

    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _load_vggt_state_dict(vggt_model: nn.Module, checkpoint, source_name: str):
    state_dict = _extract_checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported VGGT checkpoint format from {source_name}.")

    for prefix in ("module.", "geometry_encoder.vggt.", "geometry_encoder.", "vggt."):
        state_dict = _strip_state_dict_prefix(state_dict, prefix)

    incompatible_keys = vggt_model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompatible_keys, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible_keys, "unexpected_keys", []))

    if missing_keys:
        logger.warning("Missing %d VGGT keys when loading from %s", len(missing_keys), source_name)
    if unexpected_keys:
        logger.warning("Unexpected %d VGGT keys when loading from %s", len(unexpected_keys), source_name)


def _resolve_vggt_checkpoint_path(local_path: str):
    if not local_path:
        return None

    normalized_path = os.path.expanduser(local_path)
    if os.path.isfile(normalized_path):
        return normalized_path

    if os.path.isdir(normalized_path):
        for candidate_name in LOCAL_VGGT_CANDIDATE_FILES:
            candidate_path = os.path.join(normalized_path, candidate_name)
            if os.path.isfile(candidate_path):
                return candidate_path
        logger.warning(
            "VGGT checkpoint directory %s does not contain any of %s",
            normalized_path,
            ", ".join(LOCAL_VGGT_CANDIDATE_FILES),
        )
        return None

    logger.warning("VGGT checkpoint path %s does not exist.", normalized_path)
    return None


def _load_local_vggt_checkpoint(vggt_model: nn.Module, checkpoint_path: str):
    if checkpoint_path.endswith(".safetensors"):
        checkpoint = safe_load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    _load_vggt_state_dict(vggt_model, checkpoint, checkpoint_path)


def _load_vggt_from_pretrained(model_path: str):
    return VGGT.from_pretrained(
        model_path,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )


@dataclass
class FeatureFusionConfig:
    fusion_method: str = "gated"
    hidden_size: int = 4096
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class VGGTGeometryEncoder(nn.Module):
    def __init__(
        self,
        vggt_model_path=None,
        vggt_model_repo: str = DEFAULT_VGGT_MODEL_REPO,
        vggt_model_url: str = DEFAULT_VGGT_MODEL_URL,
        vggt_auto_download: bool = True,
    ):
        super().__init__()
        self.vggt_model_path = vggt_model_path
        self.vggt_model_repo = vggt_model_repo
        self.vggt_model_url = vggt_model_url
        self.vggt_auto_download = vggt_auto_download
        self._weights_initialized = False
        self.vggt = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )
        self.vggt.eval()
        self._freeze_vggt_model()

    @property
    def patch_size(self):
        return 14

    @property
    def feature_dim(self):
        return 2048

    def _freeze_vggt_model(self):
        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False

    def initialize_vggt_weights(self, force_reload: bool = False):
        if self._weights_initialized and not force_reload:
            return

        loaded_from = None

        if self.vggt_model_path:
            local_model_path = os.path.expanduser(self.vggt_model_path)
            if os.path.isdir(local_model_path):
                try:
                    self.vggt = _load_vggt_from_pretrained(local_model_path)
                    loaded_from = local_model_path
                except Exception as exc:
                    logger.warning("Failed to load VGGT model from local directory %s: %s", local_model_path, exc)

            if loaded_from is None:
                local_checkpoint_path = _resolve_vggt_checkpoint_path(self.vggt_model_path)
                if local_checkpoint_path is None:
                    logger.warning("VGGT checkpoint not found at %s, falling back to download.", self.vggt_model_path)
                else:
                    try:
                        _load_local_vggt_checkpoint(self.vggt, local_checkpoint_path)
                        loaded_from = local_checkpoint_path
                    except Exception as exc:
                        logger.warning("Failed to load local VGGT checkpoint %s: %s", local_checkpoint_path, exc)

        if loaded_from is None and self.vggt_auto_download and self.vggt_model_repo:
            try:
                pretrained_model = _load_vggt_from_pretrained(self.vggt_model_repo)
                self.vggt.load_state_dict(pretrained_model.state_dict(), strict=False)
                loaded_from = self.vggt_model_repo
            except Exception as exc:
                logger.warning("Failed to load VGGT weights from repo %s: %s", self.vggt_model_repo, exc)

        if loaded_from is None and self.vggt_auto_download and self.vggt_model_url:
            try:
                checkpoint = torch.hub.load_state_dict_from_url(self.vggt_model_url, map_location="cpu")
                _load_vggt_state_dict(self.vggt, checkpoint, self.vggt_model_url)
                loaded_from = self.vggt_model_url
            except Exception as exc:
                logger.warning("Failed to load VGGT weights from URL %s: %s", self.vggt_model_url, exc)

        if loaded_from is None:
            raise RuntimeError(
                "Unable to initialize VGGT weights. Provide --vggt_model_path or enable network download."
            )

        self._freeze_vggt_model()
        self._weights_initialized = True
        logger.info("Initialized VGGT weights from %s", loaded_from)

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

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000 ** omega)

        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def _get_2d_sincos_pos_embed(
        self,
        height: int,
        width: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1).to(dtype=dtype)

    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        h_grid: int,
        w_grid: int,
    ) -> torch.Tensor:
        reshape_to_grid = features_2d.dim() == 4

        if reshape_to_grid:
            batch_size, _, _, hidden_size = features_2d.shape
            features_2d_seq = features_2d.reshape(batch_size, h_grid * w_grid, hidden_size)
            features_3d_seq = features_3d.reshape(batch_size, h_grid * w_grid, hidden_size)
        else:
            features_2d_seq = features_2d
            features_3d_seq = features_3d

        query = self.norm1_query(features_2d_seq)
        key = self.norm1_key(features_3d_seq)
        value = self.norm1_value(features_3d_seq)
        pos_embed = self._get_2d_sincos_pos_embed(
            h_grid,
            w_grid,
            self.hidden_size,
            query.device,
            query.dtype,
        ).unsqueeze(0)

        query = query + pos_embed
        key = key + pos_embed
        attn_output, _ = self.cross_attention(query, key, value)
        fused = features_2d_seq + attn_output
        fused = fused + self.mlp(self.norm2(fused))

        if reshape_to_grid:
            return fused.reshape(batch_size, h_grid, w_grid, hidden_size)
        return fused


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

    @staticmethod
    def _infer_hw(seq_len: int):
        side = int(math.sqrt(seq_len))
        if side * side == seq_len:
            return side, side
        return seq_len, 1

    @staticmethod
    def _align_feature_grids(features_2d: torch.Tensor, features_3d: torch.Tensor):
        if features_2d.dim() == 4 and features_3d.dim() == 4:
            target_h, target_w = features_2d.shape[1], features_2d.shape[2]
            if features_3d.shape[1] != target_h or features_3d.shape[2] != target_w:
                features_3d = features_3d.permute(0, 3, 1, 2)
                features_3d = F.interpolate(
                    features_3d,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                features_3d = features_3d.permute(0, 2, 3, 1)
        return features_2d, features_3d

    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        features_2d, features_3d = self._align_feature_grids(features_2d, features_3d)
        is_grid_input = features_2d.dim() == 4

        if is_grid_input:
            h_grid, w_grid = features_2d.shape[1], features_2d.shape[2]
            features_2d_core = features_2d
            features_3d_core = features_3d
        else:
            h_grid, w_grid = self._infer_hw(features_2d.shape[1])
            features_2d_core = features_2d
            features_3d_core = features_3d

        if self.fusion_method == "add":
            fusion_feature = features_2d_core + features_3d_core
        elif self.fusion_method == "concat":
            fusion_feature = self.projection(
                torch.cat([self.norm1(features_2d_core), self.norm2(features_3d_core)], dim=-1)
            )
        elif self.fusion_method == "cross_attention":
            x = features_2d_core
            for block in self.cross_attn_blocks:
                x = block(x, features_3d_core, h_grid, w_grid)
            fusion_feature = x
        elif self.fusion_method == "gated":
            norm2d = self.norm1(features_2d_core)
            norm3d = self.norm2(features_3d_core)
            gate = self.gate_projection(torch.cat([norm2d, norm3d], dim=-1))
            fusion_feature = gate * norm2d + (1 - gate) * norm3d
        elif self.fusion_method == "weighted":
            weight_sum = self.weight_2d + self.weight_3d + 1e-6
            w2d = self.weight_2d / weight_sum
            w3d = self.weight_3d / weight_sum
            fusion_feature = w2d * features_2d_core + w3d * features_3d_core
        elif self.fusion_method == "only_3d":
            fusion_feature = features_3d_core
        elif self.fusion_method == "zero":
            fusion_feature = features_2d_core
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

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

        h_valid = (h_patch // merge_size) * merge_size
        w_valid = (w_patch // merge_size) * merge_size
        x = x[:, :h_valid, :w_valid, :]
        x = x.reshape(n_image, h_valid // merge_size, merge_size, w_valid // merge_size, merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        if self.merger_type == "mlp":
            x_flat = self.norm(x).view(-1, self.input_dim)
            x_flat = self.mlp(x_flat)
        else:
            x_flat = x.mean(dim=(3, 4))
            x_flat = x_flat.view(-1, dim)
            x_flat = self.mlp(x_flat)

        x = x_flat.reshape(n_image, h_valid // merge_size, w_valid // merge_size, -1)
        if target_hw is not None and (x.shape[1] != target_hw[0] or x.shape[2] != target_hw[1]):
            x = x.permute(0, 3, 1, 2)
            x = F.interpolate(
                x,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            x = x.permute(0, 2, 3, 1)
        return x
