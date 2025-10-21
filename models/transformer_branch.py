import torch
from torch import nn
import torch.nn.functional as F
from .helpers import choice_heads


# ## 2. Transformers Branch

# ### 2.1 Patching Helpers

# Patchify & Unpatchify
class Patchify(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.contiguous()
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        return x, (Hp, Wp)

class UnPatchify(nn.Module):
    def __init__(self, embed_dim, out_ch, patch_size=1, stride=None):
        super().__init__()
        stride = stride or patch_size
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch_size, stride=stride)

    def forward(self, x, hw):
        x = x.contiguous()
        B, N, D = x.shape
        Hp, Wp = hw
        x = x.transpose(1, 2).reshape(B, D, Hp, Wp)
        
        return self.proj(x)


# ## 2.2. Window Attention

# ### 2.2.1. Window partition Helpers
def window_partition(x, window_size):
    '''
    x : (B, C, H, W)
    return windows : (B*num_windows, window_size, window_size, C)
    '''
    x = x.contiguous()
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, "H, W must be a multiple of window_size"

    x = x.view(B, C, H//window_size, window_size, W//window_size, window_size)
    #  Permuting to (B, num_h, num_w, C, ws, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, C, window_size, window_size)   # (B*num_windows, C, ws, ws)
    return windows

def window_reverse(windows, windows_size, H, W):
    '''
    windows: (B* num_windows, C, ws, ws)
    returns: (B, C, H, W)
    '''
    windows = windows.contiguous()
    B =int(windows.shape[0]/ (H // windows_size * W // windows_size))
    C = windows.shape[1]
    x = windows.view(B, H//windows_size, W//windows_size, C, windows_size, windows_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()    # (B, C, num_h, ws, num_w, ws)
    x = x.view(B, C, H, W)

    return x


# ### 2.2.2. Window Attention Imp
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim= dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.relative_pos_bias_tbl = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size -1), num_heads)
        )   #(2*ws-1, 2*ws-1, heads)

        # gety pairwise relative positions
        coords_h  = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))   #(2, ws, ws)
        coords_flatten = torch.flatten(coords, 1)   # (2, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # (2, L, L)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # (L, L, 2)
        relative_coords[:, :, 0] += window_size -1
        relative_coords[:, :, 1] += window_size -1
        relative_coords[:, :, 0] *= 2*window_size -1
        relative_pos_idx = relative_coords.sum(-1)  # (L, L)
        self.register_buffer("relative_pos_idx", relative_pos_idx)

        nn.init.trunc_normal_(self.relative_pos_bias_tbl, std=0.02)

    def forward(self, x, H, W):
        B, N, C = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H, W should be a multiple of window_size"

        x_img = x.transpose(1, 2).contiguous().view(B, C, H, W)

        windows = window_partition(x_img, ws)   # (B*num_windows, C, ws, ws)
        num_win = windows.shape[0]
        windows = windows.flatten(2).transpose(1, 2)    # (B*num_windows, L, C) where L = ws*ws

        qkv = self.qkv(windows)
        qkv = qkv.reshape(num_win, -1, 3, self.num_heads, C//self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()    # (3, num_win, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (num_win, heads, L, L)

        # adding relative position bias
        relative_pos_bias = self.relative_pos_bias_tbl[
            self.relative_pos_idx.view(-1)
        ].view(ws * ws, ws *ws, -1) # (L, L, heads)

        relative_pos_bias = relative_pos_bias.permute(2, 0, 1).contiguous() # (heads, L, L)
        attn = attn + relative_pos_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(num_win, -1, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reversing windows back to img
        out = out.transpose(1, 2).contiguous().view(num_win, C, ws, ws)
        x_img = window_reverse(out, ws, H, W)   # (B, C, H, W)
        x_out = x_img.view(B, C, H * W).transpose(1, 2) # (B, N, C)

        return x_out
        # pass


# ## 2.3. Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self,embed_dim, n_heads, mlp_ratio=2.0, dropout=0.0, window_size=4, norm_eps=1e-6 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = WindowAttention(embed_dim, window_size=window_size, num_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, H, W):
        out = x + self.attn(self.norm1(x), H, W)
        x = out + self.mlp(self.norm2(out))

        return x
        # pass


# ## 2.4. Transformer Branch Imp
class TransformerBranch(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, depth=2, n_heads=4, patch_size=2, stage=3, window_size=3, prune_hidden=64):
        super().__init__()
        self.patch_embed = Patchify(in_ch, embed_dim, patch_size)
        self.depth = depth
        self.stage = stage
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, window_size=window_size) for _ in range(depth)
        ])
        # While unpatchifying we are also reducing the dim by passing the patch_size=1, stride=1
        self.unpatchify = UnPatchify(embed_dim, out_ch, patch_size=1, stride=1)

        # Learned Importance scorer: maps token -> Score
        self.importance = nn.Sequential(
            nn.Linear(embed_dim, prune_hidden),
            nn.GELU(),
            nn.Linear(prune_hidden, 1)
        )

        # Filler token
        self.register_parameter("filler_token", nn.Parameter(torch.zeros(1, 1, embed_dim)))
        nn.init.trunc_normal_(self.filler_token, std=0.02)
    
    def forward(self, x):
        seq, hw = self.patch_embed(x)
        B, N, D = seq.shape


        for i, blk in enumerate(self.blocks):
            keep_indices = None
            seq = blk(seq, hw[0], hw[1])

            # pruning
            # if self.stage == 3 and i >= 2:
            #     seq, keep_indices = self.prune_tokens(seq, keep_ratio=0.5)
            
            # Pruning at BottleNeck
            if self.stage == 0 and self.depth >=6 and i in (3 , 5):
                seq, keep_indices = self.prune_tokens(seq, keep_ratio=0.5)

            if keep_indices is not None:
                seq = self.restore_tokens(seq, keep_indices, N)
        
        
        out = self.unpatchify(seq, hw)

        return out

    
    def prune_tokens(self, x, keep_ratio=1.0):
        B, N, D = x.shape
        if keep_ratio <= 0.0 or keep_ratio >=1.0:
            return x, None
        
        keep_k = max(1, int(N * keep_ratio))
        scores = self.importance(x).squeeze(-1)     # (B, N)
        topk = torch.topk(scores, keep_k, dim=1, largest=True, sorted=False).indices    # (B, keep_k)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
        pruned = x[batch_idx, topk]

        return pruned, topk
    
    def restore_tokens(self, x_pruned, keep_indices, N):
        B, K, D = x_pruned.shape
        device, dtype = x_pruned.device, x_pruned.dtype
        filler = self.filler_token.expand(B, N, D).to(device=device,dtype=dtype).clone()
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)   # [B, K]
        filler[batch_idx, keep_indices] = x_pruned
        return filler