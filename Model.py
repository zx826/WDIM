import math
from dataclasses import dataclass
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
# from Resmod import Resmod
from enhance import HFNet
from mamba_ssm.modules.mamba2 import Mamba2

@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步长嵌入
        :param t: a 1-D Tensor of N indices, one per batch element.These may be fractional.param t：由 N 个索引组成的一维张量，每个批元素一个索引。 这些指数可以是小数。
        :param dim: the dimension of the output.输出的维度。
        :param max_period: controls the minimum frequency of the embeddings.控制嵌入的最小频率。
        :return: an (N, D) Tensor of positional embeddings.位置嵌入的 (N, D) 张量。
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

def modulate(x, shift, scale):
    # print(x.shape,shift.shape,scale.shape)
    return x * (1 + scale) + shift

class Unpatchify(nn.Module):
    def __init__(self, patch_size, out_channels):
        super(Unpatchify, self).__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels


    def forward(self, x):
        # x: (N, T, patch_size**2 * C)
        # 重构图像
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
        
class Mamba(nn.Module):
    def __init__(self, config: MambaConfig,
                 input_size=32,
                 patch_size=2,
                 in_channels=3,
                 ):
        super().__init__()
        self.input_size = input_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        self.y_embedder = PatchEmbed(input_size, patch_size, in_channels, config.d_model,bias=True)  # PatchEmbed函数就是patchify化,img_size是图像尺寸patch_size是每个patch的大小
        self.t_embedder = TimestepEmbedder(config.d_model)  #上面定义的时间嵌入函数
        # self.label_embedder = LabelEmbedder(num_classes, config.d_model, class_dropout_prob)
        self.config = config
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.enhance1 = HFNet(3, 64)
        self.enhance2 = HFNet(3, 64)
        self.enhance3 = HFNet(3, 64)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.out_channels = in_channels * 3
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, config.d_model), requires_grad=False)
        self.pos_embed_y = nn.Parameter(torch.zeros(1, self.y_embedder.num_patches, config.d_model), requires_grad=False)
        # Initialize (and freeze) pos_embed by sin-cos embedding:

        self.unpatchify = Unpatchify(self.x_embedder.patch_size[0],self.out_channels)
        self.final_layer = FinalLayer(config.d_model, patch_size, self.out_channels)
        self.initialize_weights()

    '''初始化权重'''
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))
        pos_embed_y = get_2d_sincos_pos_embed(self.pos_embed_y.shape[-1], int(self.y_embedder.num_patches ** 0.5))
        self.pos_embed_y.data.copy_(torch.from_numpy(pos_embed_y).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        w = self.y_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.y_embedder.proj.bias, 0)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):


        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # Initialize label embedding table:
        # nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t , y ,H1,H2,H3,pred_xstart=None,model_zero=None,**kwargs):


        x = self.x_embedder(x) + self.pos_embed_x  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        y = self.y_embedder(y) + self.pos_embed_y  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        t = self.t_embedder(t)  # (N,D),torch.Size([10, 16])

        h1 = self.enhance1(H1)
        h2 = self.enhance2(H2)
        h3 = self.enhance3(H3)

        for layer in self.layers:
            x = layer(x,t,y)

        x = self.final_layer(x,t,y)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x, h1, h2, h3

    def step(self, x, caches):


        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches



class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig ,
                       mlp_ratio=4.0):
        super().__init__()
        self.Mamba = Mamba2(config.d_model,rmsnorm=False)#,rmsnorm=False
        # self.norm = RMSNorm(config.d_model)
        self.norm = nn.LayerNorm(config.d_model, elementwise_affine=False, eps=1e-6)


        # mlp_hidden_dim = int(config.d_model * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        )
    
        self.initial()

    def initial(self):
        # Zero-out adaLN modulation layers in blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(self, x, t,y):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(t).chunk(3, dim=1)
        shift_msay, scale_msay, gate_msay = self.adaLN_modulation(y).chunk(3, dim=2)

        shift_msa = shift_msa.unsqueeze(1) + shift_msay
        scale_msa = scale_msa.unsqueeze(1) + scale_msay
        gate_msa = gate_msa.unsqueeze(1) + gate_msay
        output = gate_msa * self.Mamba(modulate(self.norm(x), shift_msa,  scale_msa)) + x
        # output = gate_msa * self.Mamba(modulate(x, shift_msa,  scale_msa)) + x





        return output


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.adaLN_modulationy = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t,y):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        shifty, scaley = self.adaLN_modulationy(y).chunk(2, dim=2)
        shift = shift.unsqueeze(1) + shifty
        scale = scale.unsqueeze(1) + scaley
        x = modulate(self.norm_final(x), shift, scale)
        # x = modulate(x, shift, scale)
        x = self.linear(x)
        return x     #  (N, T, patch_size ** 2 * out_channels)



# # taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5):
#         super().__init__()

#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model))

#     def forward(self, x):
#         output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

#         return output








def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

