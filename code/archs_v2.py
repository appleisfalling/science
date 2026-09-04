# -*- coding: utf-8 -*-
"""ACRE-C v1: Commit-or-Abstain Consensus Routing of Evidence on top of U-KAN (SPEC_v1_final).

Modules (each created only under its switch; all off == archs.UKAN, bit-identical incl. state_dict keys):
  A  CommitController  (use_ctrl)      t3 [B,128,H/8,W/8] -> polarity logit p and masses m = (m_fg, m_bg, m_a) on the simplex
                                       q = sigmoid(p) (supervised vs 8x8 occupancy);  mu = 1-|2q-1| commitment margin;
                                       rho = sigmoid(a) abstention fraction of the margin (NO target: routed gradient only);
                                       m_a = rho*mu, m_fg = q - m_a/2, m_bg = 1-q - m_a/2   (=> m_fg + m_a/2 == q by construction)
  B  ConsensusRouter   (use_fusion)    replaces the 4 `torch.add(up, skip)` fusions: fixed binomial low/high-pass skip split with
                                       state-set high-pass trust in (1/2, 3/2); polarity-partitioned normalised-convolution consensus
                                       m± (softmax kernel, eps = 5% of the window); two zero-init experts routed by m_a:
                                       out = u + (1-m_a)*f_c(m_own-u) + m_a*f_a([m+-u, m--u, (G+-u, G--u)])
  C  image-wide consensus (use_context)  G± = commitment-weighted global means of u, extra contrasts for the abstain expert (all scales)
  D  objective         (use_objective)   loss only, see losses_v2.py

Parameter names deliberately contain neither 'layer' nor 'fc' (train scripts route names with both to the KAN lr 1e-2).
"""
import torch
from torch import nn
import torch.nn.functional as F

from archs import UKAN


# ---------------------------------------------------------------- Module A
class CommitController(nn.Module):
    """t3 [B,in_c,h,w] -> (p [B,1,h,w] polarity logit, masses [B,3,h,w] = (fg, bg, abstain))."""

    def __init__(self, in_c=128, hid=64, abst='learned', q_grad=0.0):
        super().__init__()
        assert abst in ('learned', 'margin', 'half', 'none')
        self.abst, self.q_grad = abst, float(q_grad)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_c, hid, 3, padding=1), nn.BatchNorm2d(hid), nn.GELU(),
            nn.Conv2d(hid, hid, 5, padding=4, dilation=2, groups=hid), nn.BatchNorm2d(hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid), nn.BatchNorm2d(hid), nn.GELU())
        self.head = nn.Conv2d(hid, 2 if abst == 'learned' else 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # q = rho = 1/2 -> m = (1/4, 1/4, 1/2) exactly at init

    def forward(self, t3):
        z = self.head(self.trunk(t3))
        p = z[:, 0:1]
        q = torch.sigmoid(p)
        q_r = q.detach() + self.q_grad * (q - q.detach())  # polarity gradient scale (0 = anchored to L_ctrl only)
        mu = 1 - (2 * q_r - 1).abs()
        if self.abst == 'learned':
            rho = torch.sigmoid(z[:, 1:2])
        else:
            rho = {'margin': 1.0, 'half': 0.5, 'none': 0.0}[self.abst]
        m_a = rho * mu
        masses = torch.cat([q_r - 0.5 * m_a, (1 - q_r) - 0.5 * m_a, m_a], 1)
        return p, masses


def resample_masses(m, size):
    """3-mass field -> target spatial size; linear ops keep rows on the simplex (and the readout q)."""
    size = tuple(int(s) for s in size)
    if tuple(m.shape[-2:]) == size:
        return m
    if m.shape[-1] > size[-1]:
        return F.avg_pool2d(m, m.shape[-1] // size[-1])  # h/2 (s4)
    return F.interpolate(m, size=size, mode='bilinear', align_corners=False)  # 2h, 4h (s2, s1)


def _binomial5():
    b = torch.tensor([1., 4., 6., 4., 1.]) / 16.0
    return torch.outer(b, b)  # 5x5, sums to 1


# ---------------------------------------------------------------- Module B (+C)
class ConsensusRouter(nn.Module):
    def __init__(self, C, k=5, hid=None, use_global=False, detail=True, eps=0.05, eps_g=1.0):
        super().__init__()
        self.C, self.k, self.eps, self.eps_g = C, k, eps, eps_g
        self.use_global, self.detail = use_global, detail
        hid = hid or C
        # consensus kernel: softmax over k*k taps (non-negative, sum 1), log-Gaussian init with sigma = k/4
        r = torch.arange(k, dtype=torch.float32) - k // 2
        logits = -(r[:, None] ** 2 + r[None, :] ** 2) / (2 * (k / 4.0) ** 2)
        self.theta = nn.Parameter(logits[None, None])  # [1,1,k,k]
        if detail:
            self.register_buffer('lowpass', _binomial5()[None, None].repeat(C, 1, 1, 1))  # fixed, 0 params
            self.beta = nn.Parameter(torch.zeros(3, C, 1, 1))  # zero-init -> w_hi == 1
        n_a = 4 if use_global else 2
        self.commit_mix = nn.Sequential(nn.Conv2d(C, hid, 1), nn.GELU(), nn.Conv2d(hid, C, 1))
        self.abstain_mix = nn.Sequential(nn.Conv2d(n_a * C, hid, 1), nn.GELU(), nn.Conv2d(hid, C, 1))
        for mix in (self.commit_mix, self.abstain_mix):  # last layer zero-init -> out == u at init
            nn.init.zeros_(mix[-1].weight)
            nn.init.zeros_(mix[-1].bias)

    def kernel(self):
        return torch.softmax(self.theta.flatten(), 0).view_as(self.theta)  # [1,1,k,k], sum 1

    def nconv(self, x, w):
        """x [B,C,H,W], w [B,1,H,W] in [0,1] -> convex combination of x over committed neighbours;
        den in [0,1] is the committed fraction of the window; falls back to x continuously."""
        K = self.kernel()
        pad = self.k // 2
        num = F.conv2d(x * w, K.expand(self.C, 1, -1, -1).contiguous(), padding=pad, groups=self.C)
        den = F.conv2d(w, K, padding=pad)
        return (num + self.eps * x) / (den + self.eps)

    def gmean(self, x, w):
        """image-wide commitment-weighted mean with a one-pixel fallback to x (contrast -> 0 if no mass)."""
        num = (x * w).sum((2, 3), keepdim=True)
        den = w.sum((2, 3), keepdim=True)
        return (num + self.eps_g * x) / (den + self.eps_g)

    def forward(self, up, skip, masses):
        M = resample_masses(masses, up.shape[-2:])
        m_fg, m_bg, m_a = M[:, 0:1], M[:, 1:2], M[:, 2:3]
        q = m_fg + 0.5 * m_a
        if self.detail:
            # replicate-pad so the fixed filter stays a convex combination at the border too (in-image mass == 1)
            s_lo = F.conv2d(F.pad(skip, (2, 2, 2, 2), mode='replicate'), self.lowpass, groups=self.C)
            s_hi = skip - s_lo
            w_hi = 1 + 0.5 * torch.tanh(self.beta[0] * m_fg + self.beta[1] * m_bg + self.beta[2] * m_a)  # (1/2, 3/2)
            u = up + s_lo + w_hi * s_hi  # == up + skip at init
        else:
            u = up + skip
        m_pos, m_neg = self.nconv(u, m_fg), self.nconv(u, m_bg)
        m_own = q * m_pos + (1 - q) * m_neg
        contrasts = [m_pos - u, m_neg - u]
        if self.use_global:
            contrasts += [self.gmean(u, m_fg) - u, self.gmean(u, m_bg) - u]
        e_c = self.commit_mix(m_own - u)
        e_a = self.abstain_mix(torch.cat(contrasts, 1))
        return u + (1 - m_a) * e_c + m_a * e_a


# ---------------------------------------------------------------- integration
class UKAN_V2(UKAN):
    """archs.UKAN with optional ACRE-C modules. Subclass => identical module names / state_dict when all switches are off.
    Output: logits, or (logits, (p, masses)) when the controller exists."""

    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, img_size=256,
                 embed_dims=(128, 160, 256), no_kan=False, kan_mask='K,K,K',
                 use_ctrl=False, use_fusion=False, use_context=False, use_objective=False,
                 ctrl_hid=64, abst='learned', q_grad=0.0, route_detail=True, route_k=(7, 7, 5, 5),
                 route_eps=0.05, **kwargs):
        embed_dims = list(embed_dims)
        super().__init__(num_classes, input_channels, deep_supervision, img_size=img_size,
                         embed_dims=embed_dims, no_kan=no_kan, kan_mask=kan_mask, **kwargs)
        if use_context:
            use_fusion = True  # context enters only through the routers
        if use_fusion or use_objective:
            use_ctrl = True  # the controller is the shared routing state
        self.use_ctrl, self.use_fusion, self.use_context, self.use_objective = use_ctrl, use_fusion, use_context, use_objective
        if use_ctrl:
            self.ctrl = CommitController(embed_dims[0], hid=ctrl_hid, abst=abst, q_grad=q_grad)
        if use_fusion:
            C = embed_dims[0]
            kw = dict(use_global=use_context, detail=route_detail, eps=route_eps)
            self.route1 = ConsensusRouter(C // 8, k=route_k[0], hid=2 * (C // 8), **kw)  # 16  @ H/2
            self.route2 = ConsensusRouter(C // 4, k=route_k[1], hid=2 * (C // 4), **kw)  # 32  @ H/4
            self.route3 = ConsensusRouter(C, k=route_k[2], hid=C, **kw)                  # 128 @ H/8
            self.route4 = ConsensusRouter(embed_dims[1], k=route_k[3], hid=embed_dims[1], **kw)  # 160 @ H/16
        for n, _ in self.named_parameters():
            nl = n.lower()
            assert not ('layer' in nl and 'fc' in nl) or n.startswith(('block', 'dblock')), \
                f'parameter {n} would be caught by the KAN-lr rule'

    def _fuse(self, name, up, skip, masses):
        if self.use_fusion:
            return getattr(self, name)(up, skip, masses)
        return torch.add(up, skip)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = out

        p = masses = None
        if self.use_ctrl:
            p, masses = self.ctrl(t3)

        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = self._fuse('route4', out, t4, masses)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = self._fuse('route3', out, t3, masses)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = self._fuse('route2', out, t2, masses)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = self._fuse('route1', out, t1, masses)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        logits = self.final(out)
        return (logits, (p, masses)) if masses is not None else logits


def build_from_config(config):
    """Factory used by train_v2.py / eval_ckpt.py. config is the argparse dict (+ config['opt'] for extras)."""
    opt = config.get('opt', {}) or {}
    rk = opt.get('route_k', '7,7,5,5')
    rk = tuple(int(v) for v in str(rk).split(',')) if isinstance(rk, str) else (int(rk),) * 4
    return UKAN_V2(
        num_classes=config['num_classes'], input_channels=config['input_channels'],
        deep_supervision=config.get('deep_supervision', False), img_size=config.get('input_h', 256),
        embed_dims=config['input_list'], no_kan=config.get('no_kan', False), kan_mask=config.get('kan_mask', 'K,K,K'),
        use_ctrl=config.get('use_ctrl', False), use_fusion=config.get('use_fusion', False),
        use_context=config.get('use_context', False), use_objective=config.get('use_objective', False),
        ctrl_hid=int(opt.get('ctrl_hid', 64)), abst=str(opt.get('abst', 'learned')), q_grad=float(opt.get('q_grad', 0.0)),
        route_detail=bool(opt.get('route_detail', True)), route_k=rk, route_eps=float(opt.get('route_eps', 0.05)))
