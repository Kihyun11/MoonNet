import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # keep if not imported from elsewhere
    if p is None:
        p = (k - 1) // 2 * d
    return p

class Conv_SEBlock(nn.Module):
    """Conv + BN + Act + SE (residual, identity-safe)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # --- SE: channel MLP ---
        m = max(8, c2 // max(1, reduction))
        self.se_squeeze = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(c2, m, 1, bias=True)
        self.se_fc2 = nn.Conv2d(m, c2, 1, bias=True)

        # identity-safe init: last projection -> zeros => tanh(0)=0 => gate=1
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.se_fc2.bias)

    def _gate(self, x):
        y = self.se_squeeze(x)
        y = F.relu(self.se_fc1(y), inplace=True)
        z = self.se_fc2(y)               # pre-activation
        a = torch.tanh(z)                # [-1, 1]
        return x * (1.0 + a)             # residual gate

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self._gate(x)

    # used after BN fusion
    def forward_fuse(self, x):
        x = self.act(self.conv(x))
        return self._gate(x)

class Conv_CBAM(nn.Module):
    """Conv + BN + Act + CBAM (residual, identity-safe)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, reduction=16, sa_kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # --- Channel attention (shared MLP over GAP+GMP) ---
        m = max(8, c2 // max(1, reduction))
        self.ca_gap = nn.AdaptiveAvgPool2d(1)
        self.ca_gmp = nn.AdaptiveMaxPool2d(1)
        self.ca_fc1 = nn.Conv2d(c2, m, 1, bias=True)
        self.ca_fc2 = nn.Conv2d(m, c2, 1, bias=True)

        # --- Spatial attention ---
        pad = sa_kernel // 2
        self.sa_conv = nn.Conv2d(2, 1, sa_kernel, padding=pad, bias=True)

        # identity-safe init (both final projections)
        nn.init.zeros_(self.ca_fc2.weight)
        nn.init.zeros_(self.ca_fc2.bias)
        nn.init.zeros_(self.sa_conv.weight)
        nn.init.zeros_(self.sa_conv.bias)

    def _channel_gate(self, x):
        # shared MLP on GAP and GMP; sum logits
        z_avg = self.ca_fc2(F.relu(self.ca_fc1(self.ca_gap(x)), inplace=True))
        z_max = self.ca_fc2(F.relu(self.ca_fc1(self.ca_gmp(x)), inplace=True))
        z = z_avg + z_max
        a = torch.tanh(z)          # [-1, 1]
        return x * (1.0 + a)       # residual gate

    def _spatial_gate(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        z = self.sa_conv(torch.cat([avg, mx], dim=1))
        a = torch.tanh(z)          # [-1, 1]
        return x * (1.0 + a)       # residual gate

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self._channel_gate(x)
        x = self._spatial_gate(x)
        return x

    # if you also need a fuse-path:
    def forward_fuse(self, x):
        x = self.act(self.conv(x))
        x = self._channel_gate(x)
        x = self._spatial_gate(x)
        return x