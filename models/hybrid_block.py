import torch
from torch import nn
import torch.nn.functional as F
import math
from .conv_blocks import DepthWiseSeperableConv, get_norm
from .transformer_branch import TransformerBranch
from .helpers import choice_heads

# # 3. Hybrid Block

# ## 3.1. GUMBEL GATE
class GumbelGate(nn.Module):
    def __init__(self, dim, num_choices=3, init_temperature=10.0, drop_prob=0.0, learnable=False):
        super().__init__()
        self.fc = nn.Linear(dim, num_choices)
        self.num_choices = num_choices
        self.drop = drop_prob

        if learnable:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(init_temperature)))

        # Starting with zero so that all the choices are given opportunity to explore in the initial stages
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, fallback_choice=2):
        pooled = F.adaptive_avg_pool2d(x, 1)
        pooled = pooled.view(x.shape[0], -1)
        logits = self.fc(pooled)

        #Clamping the logits to prevent exploding varients
        logits = torch.clamp(logits, -15, 15)

        temperature = torch.clamp(torch.exp(self.log_temperature), min=0.5, max=10.0)

        if self.training:
            # Soft amd hard samples from gumbel softmax
            soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
            hard = F.gumbel_softmax(logits, tau=temperature, hard=True)
            
            if self.drop > 0.0:
                soft = F.dropout(soft, p=self.drop, training=True)
                soft_sum = soft.sum(dim=-1, keepdim=True)
                soft = torch.where(soft_sum > 0, soft / (soft_sum + 1e-6), torch.ones_like(soft)/self.num_choices)
            
        else: 
            # Inference Mode: Deterministic routing
            soft = F.softmax(logits, dim=-1)
            # soft = None
            idx = logits.argmax(dim=-1)
            hard = F.one_hot(idx, num_classes=self.num_choices).float()
        
        return {"soft":soft, "hard":hard}
    
    def set_temperature(self, tau):
        '''
        Manually adjusting tau (for annealing schedule)
        '''
        self.log_temperature.data.fill_(math.log(tau))


# ## 3.2. Hybrid Block with gate Imp
class HybridBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stage=3, transformer_depth=2, num_choices=3, window_size=4, norm="group"):
        super().__init__()
        self.conv = nn.Sequential(
            DepthWiseSeperableConv(in_ch, out_ch, stride=2, norm=norm),
            DepthWiseSeperableConv(out_ch, out_ch, norm=norm)
        )

        num_heads = choice_heads(out_ch, out_ch // 16)
        self.transformer = TransformerBranch(in_ch, out_ch, embed_dim=out_ch, depth=transformer_depth, n_heads=num_heads, patch_size=2, stage=stage, window_size=window_size)

        self.gate = GumbelGate(in_ch, num_choices)
        
        # learnable fusion coefficient alpha    # (1,out_ch,1,1)
        self.alpha_i = nn.Parameter(torch.zeros(out_ch)) 
        nn.init.constant_(self.alpha_i, 0.0)
        
        # self.alpha = nn.Parameter(torch.full((1, out_ch, 1, 1), 0.5), requires_grad=True) # scalar fuse weight

        self.fusion_norm = get_norm(norm, out_ch)
        self.fusion_act = nn.ReLU(inplace=True)
        self.num_choices = num_choices

    @property
    def alpha(self):
        '''Mapping the aplha to [0.1, 0.9]'''
        a = 0.1 + 0.8 * torch.sigmoid(self.alpha_i)
        return a.view(1, -1, 1, 1)

    def forward(self, x):
        '''
        Returns (out, gates) during training (out is weighted by hard decisions if training),
        For inference (deploy True or model.eval()), does batched grouping to avoid per loop
        Note: returning gates are dict(), if using Parallel GPUs consider using a aggregate function to detach from the device
        '''

        gates = self.gate(x)        # gates["hard"], gates["soft"] shape (B, num_choices=3)
        gate_soft = gates["soft"]
        gate_hard = gates["hard"]
        gate = gate_soft + (gate_hard - gate_soft).detach()

        if self.training:
            # Computing both branches (batch wise)
            c = self.conv(x)    # out shape: (B, C, H`, W`)
            t = self.transformer(x) # out shape: (B, C, H`, W`)

            # Channel wise fusion
            both = self.alpha * c + (1.0 - self.alpha) * t
            both = self.fusion_act(self.fusion_norm(both)) 
            
            # building stacked for weights sum broadcasting
            stacked = torch.stack([c, t, both], dim=1)  # (B, 3, C, H, W)
            gate_w = gate.view(x.shape[0], self.num_choices, 1, 1, 1)
            out = (stacked * gate_w).sum(dim=1)
            out = out.contiguous()
            return out, gates

        else:
            # Inference: (using effecient routing)
            with torch.no_grad():
                # self.alpha.clamp_(0.1, 0.9)
                device = x.device
                probs = gates["hard"]   # (B, num_choices) one-hot
                B = x.shape[0]

                decisions = probs.argmax(dim=-1).to(device).long()   # ENsuring 1-D long tensor

                # Grouping samples by decision
                groups = [torch.where(decisions == i)[0] for i in range(self.num_choices)]
                
                # precomputing sample to infer shape from conv branch (cheaper)
                sample = x[:1]
                c_sample = self.conv(sample)      # (1, C_out, H', W')
                out_ch, out_h, out_w = c_sample.shape[1], c_sample.shape[2], c_sample.shape[3]

                # Preallocate output with explicit device/dtype and contiguous memory
                outs = torch.zeros((B, out_ch, out_h, out_w), device=device, dtype=c_sample.dtype).contiguous()

                # Helper to insert a source tensor into outs at indices
                def safe_index_copy(out_tensor, idx, src):
                    nonlocal outs
                    if src.dtype != outs.dtype:
                        outs = outs.to(dtype=src.dtype, device=src.device)
                    
                    outs.index_copy_(0, idx, src)

                # Processing group 0 : CONV ONLY
                if groups[0].numel() > 0:
                    idx = groups[0]
                    safe_index_copy(outs, idx, self.conv(x[idx]))
                
                # Processing group 1: TRANSFORMER ONLY
                if groups[1].numel() > 0:
                    idx = groups[1]
                    safe_index_copy(outs, idx, self.transformer(x[idx]))

                # Group 2: Both fused
                if groups[2].numel() > 0:
                    idx = groups[2]
                    c = self.conv(x[idx])
                    t = self.transformer(x[idx])
                    
                    fusion = self.alpha * c + (1.0 - self.alpha) * t
                    fused = self.fusion_act(self.fusion_norm(fusion))
                    fused = fused.contiguous()
                    safe_index_copy(outs, idx, fused)
                
                outs = outs.contiguous()

                return outs, gates
