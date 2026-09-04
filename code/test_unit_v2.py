# -*- coding: utf-8 -*-
"""ACRE-C v1 (archs_v2 / losses_v2) unit checks. Run on CPU or GPU; no dataset needed.
a) all switches off: state_dict keys identical to archs.UKAN and forward bit-identical (max|diff| == 0)
b) identity at init: with fusion(+context) on and baseline weights loaded, output == baseline (zero-init experts, beta = 0)
   masses at init == (1/4, 1/4, 1/2) exactly; simplex; m_fg + m_a/2 == sigmoid(p) by construction
c) shapes / params for every switch combination and ablation (abst, route_detail); budget <= 7.0 M
d) KAN-lr grouping catches exactly the same tensors as the baseline; no-wd group = .theta/.beta only
e) dL_ctrl/da == 0 by construction; with q_grad = 0 the segmentation gradient reaches a but not p
f) focus loss: all-ones target => z_gt == 0 everywhere (count_include_pad=False); weight cap
g) router numerics: kernel sums to 1; nconv / gmean with zero mass == identity
h) gradient flows into every new parameter after one step; two optimizer steps with the criterion
i) optional calflops GFLOPs (if installed) -- expect ~6.18 G for full, budget <= 7.0
"""
import torch
import torch.nn.functional as F
import archs
import archs_v2
import losses_v2

torch.manual_seed(0)
EMB = [128, 160, 256]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(2, 3, 256, 256, device=dev)
y = (torch.rand(2, 1, 256, 256, device=dev) > 0.5).float()
y[:, :, :128] = 1.0  # half pure-fg, half mixed

base = archs.UKAN(num_classes=1, input_channels=3, embed_dims=EMB).to(dev).eval()
n_base = sum(p.numel() for p in base.parameters())


def cfg(**kw):
    c = dict(num_classes=1, input_channels=3, input_list=EMB, input_h=256, opt={})
    c.update(kw)
    return c


# a) bit identity, all off
v2 = archs_v2.build_from_config(cfg()).to(dev)
assert list(v2.state_dict().keys()) == list(base.state_dict().keys()), 'state_dict keys differ'
v2.load_state_dict(base.state_dict(), strict=True)
v2.eval()
with torch.no_grad():
    yb = base(x)
    yv = v2(x)
assert not isinstance(yv, tuple)
d = (yb - yv).abs().max().item()
print(f'[a] all-off: max|diff| = {d:.3e}, n_params = {n_base}')
assert d == 0.0, 'all-off path is not bit-identical'

# b) identity at init for fusion(+context), masses init
for name, kw in [('fusion', dict(use_fusion=True)), ('fusion+context', dict(use_context=True)),
                 ('fusion nodetail', dict(use_fusion=True, opt=dict(route_detail=False))),
                 ('full abst=margin', dict(use_context=True, use_objective=True, opt=dict(abst='margin')))]:
    m = archs_v2.build_from_config(cfg(**kw)).to(dev)
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    m.eval()
    with torch.no_grad():
        logits, (p, masses) = m(x)
    d = (logits - yb).abs().max().item()
    mf, mb, ma = masses[:, 0].mean().item(), masses[:, 1].mean().item(), masses[:, 2].mean().item()
    print(f'[b] {name}: identity-at-init max|diff| = {d:.3e}; init masses fg/bg/a = {mf:.3f}/{mb:.3f}/{ma:.3f}')
    assert d < 1e-4, 'router not identity at init'
    assert torch.allclose(masses.sum(1), torch.ones_like(masses[:, 0]), atol=1e-6)
    assert torch.allclose(masses[:, 0] + 0.5 * masses[:, 2], torch.sigmoid(p[:, 0]), atol=1e-6)
    if 'margin' not in name:
        assert abs(mf - 0.25) < 1e-6 and abs(ma - 0.5) < 1e-6, 'init masses are not (1/4,1/4,1/2)'
    else:
        assert abs(ma - 1.0) < 1e-6  # rho == 1, q == 1/2 -> mu == 1

# c) shapes / params
for name, kw in [('ctrl', dict(use_ctrl=True)), ('ctrl+fusion', dict(use_fusion=True)),
                 ('ctrl+fusion+context', dict(use_context=True)),
                 ('full', dict(use_ctrl=True, use_fusion=True, use_context=True, use_objective=True)),
                 ('ctrl+objective (loss-only)', dict(use_objective=True)),
                 ('full route_detail=0', dict(use_context=True, use_objective=True, opt=dict(route_detail=False))),
                 ('full abst=none', dict(use_context=True, use_objective=True, opt=dict(abst='none')))]:
    m = archs_v2.build_from_config(cfg(**kw)).to(dev).eval()
    with torch.no_grad():
        logits, (p, masses) = m(x)
    assert logits.shape == (2, 1, 256, 256) and masses.shape == (2, 3, 32, 32) and p.shape == (2, 1, 32, 32)
    assert (masses >= -1e-6).all() and ((masses.sum(1) - 1).abs() < 1e-5).all()
    n = sum(q.numel() for q in m.parameters())
    print(f'[c] {name}: params {n/1e6:.4f}M (+{n-n_base})')
    assert n <= 7.0e6

# d) KAN-lr rule + no-wd group
full = archs_v2.build_from_config(cfg(use_context=True, use_objective=True)).to(dev)
kan_base = {n for n, _ in base.named_parameters() if 'layer' in n.lower() and 'fc' in n.lower()}
kan_v2 = {n for n, _ in full.named_parameters() if 'layer' in n.lower() and 'fc' in n.lower()}
assert kan_base == kan_v2, 'new parameter caught by KAN-lr rule'
nowd = [n for n, _ in full.named_parameters() if n.endswith(('.theta', '.beta'))]
assert len(nowd) == 8 and not (set(nowd) & kan_v2)
print(f'[d] KAN-lr group unchanged: {len(kan_v2)} tensors; no-wd group: {len(nowd)} tensors')
ctrl_p = sum(q.numel() for n, q in full.named_parameters() if n.startswith('ctrl.'))
route_p = {s: sum(q.numel() for n, q in full.named_parameters() if n.startswith(f'route{s}.')) for s in (1, 2, 3, 4)}
print(f'[d] ctrl params {ctrl_p}; router params {route_p}; total added {ctrl_p + sum(route_p.values())}')
# SPEC_v1 §3 figures (review R11: assert, do not just print)
assert ctrl_p == 76610 and route_p == {1: 3777, 2: 14673, 3: 115609, 4: 180345}, (ctrl_p, route_p)
assert ctrl_p + sum(route_p.values()) == 391014

# e) gradient decoupling
full.train()
lg, (p, masses) = full(x)
Lc = losses_v2.commit_loss(p, y)
g = torch.autograd.grad(Lc, full.ctrl.head.weight)[0]
print(f'[e] dL_ctrl/d(head): row p |g| = {g[0].abs().sum():.3e}, row a |g| = {g[1].abs().sum():.3e} (must be 0)')
assert g[0].abs().sum() > 0 and g[1].abs().sum() == 0
with torch.no_grad():  # perturb experts so a routing gradient exists
    for r in ('route1', 'route2', 'route3', 'route4'):
        mod = getattr(full, r)
        mod.commit_mix[-1].weight.normal_(0, 0.05)
        mod.abstain_mix[-1].weight.normal_(0, 0.05)
    full.ctrl.head.weight.normal_(0, 0.05)  # non-zero head so the a-row can pass gradient into the trunk
lg, (p, masses) = full(x)
Lseg = F.binary_cross_entropy_with_logits(lg, y)
g = torch.autograd.grad(Lseg, full.ctrl.head.weight, retain_graph=True)[0]
print(f'[e] dL_seg/d(head) q_grad=0: row p |g| = {g[0].abs().sum():.3e} (expect 0), row a |g| = {g[1].abs().sum():.3e} (>0)')
# NB (review R5): only the *direct* route dL_seg/dp is zero; ctrl.trunk.* still receives L_seg gradient through the
# a-row (shared trunk), so polarity can drift indirectly -- that is what gate D1 monitors during training.
trunk_params = [q for n, q in full.named_parameters() if n.startswith('ctrl.trunk.') and q.requires_grad]
gt = torch.autograd.grad(Lseg, trunk_params, allow_unused=True)
assert any(gi is not None and gi.abs().sum() > 0 for gi in gt), 'trunk expected to receive L_seg gradient via a-row'
assert g[0].abs().sum() == 0 and g[1].abs().sum() > 0

# f) focus loss
ones = torch.ones(2, 1, 256, 256, device=dev)
assert losses_v2.gt_band(ones).abs().max().item() == 0, 'z_gt on all-ones target must be 0 (count_include_pad)'
lf_gt = losses_v2.focus_loss(torch.zeros_like(ones), None, y, src='gt')
lf_un = losses_v2.focus_loss(torch.zeros_like(ones), masses.detach(), y, src='union')
print(f'[f] z_gt(all-ones)==0 OK; focus_loss(gt) = {lf_gt.item():.4f}, focus_loss(union) = {lf_un.item():.4f} (BCE(0)=0.693)')
assert lf_gt.item() <= 0.6932 + 1e-4 and lf_un.item() <= 0.6932 + 1e-4  # cap: never above 4x mean-weighted, here <= mean BCE
w = losses_v2.gt_band(y)
assert w[:, :, :100].abs().max().item() == 0  # pure fg region -> no band

# g) router numerics
r = full.route3
K = r.kernel()
print(f'[g] kernel sum {K.sum().item():.6f}, centre {K.flatten()[K.numel()//2].item():.3f}, corner {K[0,0,0,0].item():.4f}')
assert abs(K.sum().item() - 1) < 1e-5
u = torch.randn(1, 128, 32, 32, device=dev); w0 = torch.zeros(1, 1, 32, 32, device=dev)
assert torch.allclose(r.nconv(u, w0), u, atol=1e-5) and torch.allclose(r.gmean(u, w0), u, atol=1e-5)
print('[g] nconv/gmean with zero mass == u  OK')

# h) gradient flow + criterion, two steps
m = archs_v2.build_from_config(cfg(use_context=True, use_objective=True)).to(dev).train()
config = cfg(use_ctrl=True, use_fusion=True, use_context=True, use_objective=True)
crit = losses_v2.build_criterion(config)
opt = torch.optim.Adam(m.parameters(), lr=1e-4)
out = m(x)
loss, logits, terms = crit(out, y)
assert torch.isfinite(loss)
opt.zero_grad(); loss.backward()


def dead_params(model):
    return [n for n, q in model.named_parameters() if n.startswith(('ctrl.', 'route')) and
            (q.grad is None or q.grad.abs().sum().item() == 0)]


dead0 = dead_params(m)
print(f'[h] loss {loss.item():.4f} terms {{{", ".join(f"{k}: {v:.4f}" for k, v in terms.items())}}}')
print(f'[h] zero-grad new params at step 0: {len(dead0)} tensors (expected under zero-init: ctrl.trunk.* behind the zero head, '
      f'theta and *_mix.0.* behind the zero expert output layers)')
assert all(n.endswith('theta') or n.startswith('ctrl.trunk.') or '_mix.0.' in n for n in dead0), f'gradient starvation: {dead0}'
assert m.ctrl.head.weight.grad.abs().sum() > 0 and m.ctrl.head.bias.grad.abs().sum() > 0
for r in ('route1', 'route2', 'route3', 'route4'):
    assert getattr(m, r).commit_mix[-1].weight.grad.abs().sum() > 0 and getattr(m, r).beta.grad.abs().sum() > 0
opt.step(); opt.zero_grad()
out2 = m(x)
loss2, logits2, _ = crit(out2, y)
loss2.backward()
dead1 = dead_params(m)
print(f'[h] zero-grad new params at step 1: {dead1}')
assert not dead1, 'gradient starvation after one step'
opt.step()
print('[h] two optimizer steps OK; output changed:', (logits2 - logits).abs().max().item() > 0)

# i) calflops
try:
    from calflops import calculate_flops
    for name, kw in [('baseline', {}), ('full', dict(use_context=True, use_objective=True))]:
        mm = archs_v2.build_from_config(cfg(**kw)).to(dev).eval()
        flops, macs, params = calculate_flops(model=mm, input_shape=(1, 3, 256, 256), output_as_string=False,
                                              print_results=False, print_detailed=False)
        print(f'[i] {name}: GFLOPs {flops/1e9:.3f}  GMACs {macs/1e9:.3f}  params {params/1e6:.3f}M')
        if name == 'full':
            assert flops / 1e9 <= 7.0
except ImportError:
    print('[i] calflops not installed, skipped')
except TypeError as e:  # some calflops versions choke on tuple scale_factor in F.interpolate
    print(f'[i] calflops failed on this version ({e}); use eval_ckpt.py --flops on the server')
print('ALL V2 UNIT TESTS PASSED')
