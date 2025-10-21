import torch
from torch import nn

def get_norm(norm_type, num_ch, num_groups=8):
    '''
    Return normalized layer: GroupNorm is defaulted for small batches.
    '''
    if norm_type == "group":
        groups = min(num_groups, num_ch)
        while num_ch % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_ch)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_ch)
    else:
        return nn.Identity()

# ## 1. Depthwise seperable conv block
class DepthWiseSeperableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm="group"):
        super().__init__()
        self.out_channels = out_ch
        mid_ch = max(1, in_ch)
        self.depthwise = nn.Conv2d(in_ch, mid_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.dw_norm = get_norm(norm, mid_ch)

        self.pointwise = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.pw_norm = get_norm(norm, out_ch)

        self.act = nn.ReLU(inplace=True)

        if stride !=1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.act(self.dw_norm(out))
        out = self.pointwise(out)
        out = self.pw_norm(out)
        out = out + self.skip(x)
        out = out.contiguous()

        return self.act(out)
