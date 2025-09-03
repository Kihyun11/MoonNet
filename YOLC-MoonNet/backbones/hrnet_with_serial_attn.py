# Copyright (c) OpenMMLab.
# HRNet backbone with identity-safe, switchable per-branch attention.
# Supports: 'serial' | 'cbam' | 'se' | 'none' | **'alternate' (SE/CBAM alternating)**
# MMDetection 2.x / MMCV 1.x / PyTorch 2.0–2.1 (Python 3.10)

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.hrnet import HRNet
from mmdet.models.builder import BACKBONES


# ----------------------------
# Identity-safe attention gates
# ----------------------------

class SEGate(nn.Module):
    """Squeeze-Excite (residual, identity-safe). y = x * (1 + tanh(MLP(gap(x))))"""
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        m = max(8, c // max(1, r))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, m, 1, bias=True)
        self.fc2 = nn.Conv2d(m, c, 1, bias=True)
        # identity-safe init: last layer zeros => tanh(0)=0 => gate = 1 at init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.avg(x)
        a = F.relu(self.fc1(a), inplace=True)
        a = torch.tanh(self.fc2(a))
        return x * (1.0 + a)


class CBAMGate(nn.Module):
    """CBAM-like channel + spatial (both residual, identity-safe)."""
    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        m = max(8, c // max(1, r))
        # Channel attention (avg+max pooled descriptor)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxp = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c, m, 1, bias=True)
        self.fc2 = nn.Conv2d(m, c, 1, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        # Spatial attention
        pad = k // 2
        self.spatial = nn.Conv2d(2, 1, k, padding=pad, bias=True)
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel gate
        ca = self.avg(x) + self.maxp(x)
        ca = F.relu(self.fc1(ca), inplace=True)
        ca = torch.tanh(self.fc2(ca))
        x = x * (1.0 + ca)
        # Spatial gate
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        sa = torch.tanh(self.spatial(torch.cat([avg, mx], dim=1)))
        x = x * (1.0 + sa)
        return x


class SerialAttn(nn.Module):
    """Channel gate -> Spatial gate (residual, identity-safe)."""
    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        m = max(8, c // max(1, r))
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, m, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, c, 1, bias=True),
        )
        nn.init.zeros_(self.ca[-1].weight)
        nn.init.zeros_(self.ca[-1].bias)
        pad = k // 2
        self.sa = nn.Conv2d(2, 1, k, padding=pad, bias=True)
        nn.init.zeros_(self.sa.weight)
        nn.init.zeros_(self.sa.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = torch.tanh(self.ca(x))
        x = x * (1.0 + ca)
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        sa = torch.tanh(self.sa(torch.cat([avg, mx], dim=1)))
        x = x * (1.0 + sa)
        return x


def make_attn(kind: str, c: int, r: int = 16, k: int = 7) -> nn.Module:
    kind = (kind or 'none').lower()
    if kind == 'se':
        return SEGate(c, r)
    if kind == 'cbam':
        return CBAMGate(c, r, k)
    if kind == 'serial':
        return SerialAttn(c, r, k)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(f'Unknown attn_kind: {kind!r}. Use: se|cbam|serial|none|alternate')


def _alternating_kinds(n: int, start: str = 'se') -> List[str]:
    """Produce ['se','cbam','se','cbam',...] length n, starting with 'se' or 'cbam'."""
    a, b = ('se', 'cbam') if start.lower() == 'se' else ('cbam', 'se')
    kinds = []
    flip = False
    for i in range(n):
        kinds.append(a if not flip else b)
        flip = not flip
    return kinds


# ---------------------------------
# HRNet backbone with branch attention
# ---------------------------------

@BACKBONES.register_module()
class HRNetWithAltAttn(HRNet):
    """
    HRNet with optional per-branch attention applied to the final-stage outputs.

    Args:
        extra (dict): same as HRNet (stage configs).
        in_channels (int): input channels.
        norm_eval (bool): as in HRNet.
        with_cp (bool): as in HRNet (checkpointing).
        attn_reduction (int): reduction ratio for channel attention parts.
        attn_kernel (int): kernel size (odd) for spatial attention.
        attn_kind (str): 'serial' | 'cbam' | 'se' | 'none' | 'alternate'.
        attn_start (str): when attn_kind='alternate', which to start with ('se' or 'cbam').
        init_cfg (dict): weight init (e.g., HRNet pretrain).
    """
    def __init__(self,
                 extra,
                 in_channels: int = 3,
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 attn_reduction: int = 16,
                 attn_kernel: int = 7,
                 attn_kind: str = 'serial',
                 attn_start: str = 'se',
                 init_cfg=None):
        super().__init__(extra=extra,
                         in_channels=in_channels,
                         norm_eval=norm_eval,
                         with_cp=with_cp,
                         init_cfg=init_cfg)

        self.attn_reduction = int(attn_reduction)
        self.attn_kernel = int(attn_kernel)
        self.attn_kind = (attn_kind or 'serial').lower()
        self.attn_start = (attn_start or 'se').lower()

        # Derive branch channels from HRNet config (stage4), else fallback
        if 'stage4' in extra and 'num_channels' in extra['stage4']:
            out_chs: Sequence[int] = tuple(extra['stage4']['num_channels'])
        else:
            out_chs = (48, 96, 192, 384)

        n_branches = len(out_chs)

        # Determine attention kind per branch
        if self.attn_kind == 'alternate':
            per_branch = _alternating_kinds(n_branches, start=self.attn_start)
        else:
            per_branch = [self.attn_kind] * n_branches

        # Build one attention gate per branch
        self.branch_attn = nn.ModuleList([
            make_attn(k, c, self.attn_reduction, self.attn_kernel)
            for k, c in zip(per_branch, out_chs)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feats: List[torch.Tensor] = super().forward(x)  # HRNet outputs (list/tuple of 4)
        assert len(feats) == len(self.branch_attn), \
            f"HRNet returned {len(feats)} maps, but have {len(self.branch_attn)} attention blocks."
        outs = [attn(f) for attn, f in zip(self.branch_attn, feats)]
        return tuple(outs)
