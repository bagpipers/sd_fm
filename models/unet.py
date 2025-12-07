import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    # (変更なし)
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) 
        return embeddings

class ResBlock(nn.Module):
    # (変更なし)
    def __init__(self, in_c, out_c, time_emb_dim, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_c)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_c)
        self.time_proj = nn.Linear(time_emb_dim, out_c)
        self.residual_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, t_emb):
        h = self.silu(self.gn1(self.conv1(x)))
        t_emb_proj = self.time_proj(self.silu(t_emb))
        h = h + t_emb_proj.view(h.shape[0], h.shape[1], 1, 1)
        h = self.gn2(self.conv2(h))
        return h + self.residual_conv(x)

class SimpleCrossAttention(nn.Module):
    # (変更なし)
    def __init__(self, query_dim, context_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=context_dim,
            vdim=context_dim,
            num_heads=num_heads,
            batch_first=True 
        )
        self.norm = nn.LayerNorm(query_dim)
    def forward(self, x, context):
        B, C, H, W = x.shape
        x_seq = x.view(B, C, H * W).permute(0, 2, 1)
        attn_output, _ = self.attn(query=x_seq, key=context, value=context)
        x_seq = self.norm(x_seq + attn_output)
        x_out = x_seq.permute(0, 2, 1).view(B, C, H, W)
        return x_out


# --- メインのUNetモデル (★ 修正箇所 ★) ---

class TextConditionedUNet(nn.Module):
    """
    config.yaml の設定に基づき、
    正しい 'context_dim' を読み込んで構築されるUNet。
    """
    def __init__(self, config):
        super().__init__()
        
        data_cfg = config['data']
        model_cfg = config['model']
        unet_cfg = model_cfg['unet_config']
        
        # --- ★ 変更点 ★ ---
        # どのモードで動作しているかに基づき、
        # Cross-Attention の次元 (context_dim) を決定する
        condition_type = model_cfg['condition_type']
        if condition_type == "clip":
            context_dim = model_cfg['clip_config']['embed_dim'] # 例: 768
        elif condition_type == "class":
            context_dim = model_cfg['class_config']['embed_dim'] # 例: 512
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")
        # --- ★ 変更ここまで ★ ---

        in_channels = data_cfg['channels']       # 3
        base_dim = unet_cfg['base_dim']        # 64
        dim_mults = unet_cfg['dim_mults']      # [1, 2, 4]
        num_heads = unet_cfg['num_attn_heads']   # 4
        
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        
        # (Time Embedding)
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # (Input)
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # (Down Blocks)
        self.down_blocks = nn.ModuleList()
        num_resolutions = len(dims) - 1
        
        for i in range(num_resolutions):
            in_c = dims[i]
            out_c = dims[i+1]
            self.down_blocks.append(nn.ModuleList([
                ResBlock(in_c, out_c, time_dim),
                # ★ context_dim を渡す
                SimpleCrossAttention(out_c, context_dim, num_heads),
                nn.Conv2d(out_c, out_c, 4, stride=2, padding=1) 
            ]))
            
        # (Mid Blocks)
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = SimpleCrossAttention(mid_dim, context_dim, num_heads)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_dim)

        # (Up Blocks)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_resolutions)):
            in_c = dims[i+1]
            out_c = dims[i]
            
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
                ResBlock(out_c + in_c, out_c, time_dim),
                SimpleCrossAttention(out_c, context_dim, num_heads),
            ]))

        # (Output)
        self.final_conv = nn.Conv2d(base_dim, in_channels, 1)
        print(f"Initialized Simple UNet. Mode='{condition_type}', Cross-Attention Dim={context_dim}")

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        # (forward のロジックは変更なし)
        
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        
        skip_connections = []
        for resblock, cross_attn, downsample in self.down_blocks:
            x = resblock(x, t_emb)
            x = cross_attn(x, context)
            skip_connections.append(x)
            x = downsample(x)
            
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t_emb)
        
        
        for upsample, resblock, cross_attn in self.up_blocks:
            x = upsample(x)
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1) 
            x = resblock(x, t_emb)
            x = cross_attn(x, context)
            
        return self.final_conv(x)