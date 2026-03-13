import torch
import torch.nn as nn
import re

# ==========================================
# [新增] Evo-0 QKV 空间融合层
# ==========================================
class Evo0FusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这里的 hidden_size 通常是 LLM 的维度，比如 4096
        self.hidden_size = config.hidden_size 
        self.num_heads = getattr(config, 'qkv_num_heads', 8)
        
        # 1. Image Tokens -> Query
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 2. VGGT Latent Tokens -> Key & Value
        self.kv_proj = nn.Linear(self.hidden_size, self.hidden_size * 2)
        
        # 3. 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=self.num_heads, 
            batch_first=True
        )
        
        # 4. 归一化与 FFN
        self.ln_q = nn.LayerNorm(self.hidden_size)
        self.ln_kv = nn.LayerNorm(self.hidden_size)
        self.ln_post = nn.LayerNorm(self.hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )

    def forward(self, vision_tokens, vggt_latents):
        # vision_tokens (Query): [Batch, Seq_Vision, H]
        # vggt_latents (Key, Value): [Batch, Seq_VGGT, H]
        vision_norm = self.ln_q(vision_tokens)
        vggt_norm = self.ln_kv(vggt_latents)
        
        q = self.q_proj(vision_norm)          
        kv = self.kv_proj(vggt_norm)          
        k, v = kv.chunk(2, dim=-1)            
        
        # 用 2D 图像特征(Q)去查询 3D 空间特征(K, V)
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        
        out = q + attn_out
        out = out + self.ffn(self.ln_post(out))
        
        # 返回的 out 序列长度与 vision_tokens 完全一致！
        return out
        
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
