import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class FourierEmbedding(nn.Module):
    """
    把 xyz ∈ [-1,1]^3 编码成更高频的 2K×3 维向量:
      γ(x) = [sin(2^0 π x), cos(2^0 π x), …, sin(2^{K-1} π x), cos(2^{K-1} π x)]
    """
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        # 用 register_buffer 把 freq_bands 注册成 buffer，保证它随模型一起移动
        freq_bands = 2.0 ** torch.arange(num_freqs) * torch.pi  # (K,)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:               # (B,3)
        # 这里的 self.freq_bands 已经自动在 x.device 上了，不会再出错
        x = x.unsqueeze(-1)                                           # (B,3,1)
        embed = x * self.freq_bands                                   # (B,3,K)
        embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=-1)
        return embed.flatten(-2)  

# -------------------------------------------------------------
# 1. Decoder 网络
# -------------------------------------------------------------
class DeepSDF(nn.Module):
    """
    只训练一个 mesh 时，隐变量 z 直接作为模型参数：
      - z shape = (1, latent_dim)，每个前向都 expand 到 (B, latent_dim)
      - MLP 8×512，ReLU，单次 skip（在第 4 层拼接 [z‖xyz]）
      - 输出 tanh 约束到 [-1,1]
    """
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        skip_layer: int = 4,
        num_freqs: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.skip_layer = skip_layer

        # 1) 初始化潜变量 z，形状 (1, latent_dim)
        self.z = nn.Parameter(torch.zeros(1, latent_dim))
        nn.init.normal_(self.z, 0.0, 0.01)  # N(0, 0.01^2) 初始

        # 2) 搭建 8×512 MLP + 一次 skip
        self.embed_fn = FourierEmbedding(num_freqs=num_freqs)
        embed_dim = 3 * 2 * num_freqs
        in_ch = latent_dim + embed_dim  # 首层输入通道 = z_dim + xyz_dim
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            # 如果到了 skip 层，则再拼接一次 (z_dim + 3) 到输入通道
            if i == skip_layer:
                in_ch += latent_dim + embed_dim
            lin = weight_norm(nn.Linear(in_ch, hidden_dim))
            self.linears.append(lin)
            in_ch = hidden_dim

        self.final = weight_norm(nn.Linear(hidden_dim, 1))
        self.act = nn.ReLU()

        # 按论文推荐的初始化方式
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        输入：
          xyz : (B, 3)，已归一化到 [-1,1]
        输出：
          pred_sdf : (B, 1)，值范围 [-1,1]
        """
        B = xyz.shape[0]
        # 把 z 从 (1, latent_dim) expand 到 (B, latent_dim)
        embed_xyz = self.embed_fn(xyz)
        z_expand = self.z.expand(B, -1)       # (B, latent_dim)

        # 初始拼接 [z‖xyz]
        h = torch.cat([z_expand, embed_xyz], dim=-1)  # (B, latent_dim + 3)

        # MLP 前向，单次 skip
        for i, layer in enumerate(self.linears):
            if i == self.skip_layer:
                # 在第 4 层前再把 [z‖xyz] 拼接一次
                h = torch.cat([h, z_expand, embed_xyz], dim=-1)
            h = self.act(layer(h))

        return torch.tanh(self.final(h)), z_expand      # (B, 1)

# ------------------------- 损失函数 ----------------------------
# def deep_sdf_loss_single(pred, sdf_gt, z_expand, clamp_dist=0.1, lambda_z=1e-4):
#     pred_c   = torch.clamp(pred,  -clamp_dist, clamp_dist)
#     target_c = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
#     data_loss = F.l1_loss(pred_c, target_c)
#     reg_loss  = lambda_z * torch.mean(torch.sum(z_expand**2, dim=1))
#     return data_loss + reg_loss