# -*- coding: utf-8 -*-
"""ACRE-C v1 objective (Module D) + controller supervision, wrapped as a criterion for train_v2.py.

L = L_base + lam_ctrl * L_ctrl + lam_obj * L_foc
  L_base = BCEDiceLoss (frozen: 0.5*BCE + Dice)
  L_ctrl = BCE_with_logits(p, ybar)                         p = polarity logit, ybar = AvgPool_8(Y)  (soft cell occupancy)
           -> touches neither rho nor m_a: dL_ctrl/da == 0 by construction (unit-tested); abstention is set by routing only.
  L_foc  = sum(w * BCE_x) / max(sum(w), rho_f * N)          w = max(z_gt, stopgrad(up(m_a)))  (src='union') or z_gt (src='gt')
           z_gt = 1 - |2 * AvgPool_{k_gt, stride 1, count_include_pad=False}(Y) - 1|  (GT window mixedness; no morphology / DT)
           cap rho_f = 0.25 => the band is never up-weighted by more than 4x relative to plain mean BCE.
Terms returned for logging: ctrl, foc (when active) + diagnostics ma_mean, rho_std, ctrl_bce, cell_iou (drift/collapse gates).
"""
import torch
import torch.nn.functional as F

import losses


def cell_target(target, h):
    """[B,1,H,W] {0,1} -> soft occupancy [B,1,h,h] (exact block mean; H % h == 0)."""
    return F.avg_pool2d(target, target.shape[-1] // h)


def commit_loss(p, target):
    """p [B,1,h,w] polarity logit; target [B,1,H,W] in {0,1}."""
    return F.binary_cross_entropy_with_logits(p, cell_target(target, p.shape[-1]))


def gt_band(target, k_gt=15):
    occ = F.avg_pool2d(target, k_gt, stride=1, padding=k_gt // 2, count_include_pad=False)
    return 1 - (2 * occ - 1).abs()


def focus_loss(logits, masses, target, k_gt=15, rho_f=0.25, src='union'):
    assert src in ('union', 'gt')
    with torch.no_grad():
        w = gt_band(target, k_gt)
        if src == 'union' and masses is not None:
            m_a = F.interpolate(masses[:, 2:3], size=target.shape[-2:], mode='bilinear', align_corners=False)
            w = torch.maximum(w, m_a)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (w * bce).sum() / torch.clamp(w.sum(), min=rho_f * w.numel())


@torch.no_grad()
def controller_diagnostics(p, masses, target):
    """Gate monitors (SPEC_v1 §8): mean abstention, spread of rho (collapse to a constant => routing unused),
    controller BCE and cell-level IoU of q vs occupancy (polarity drift)."""
    ybar = cell_target(target, masses.shape[-1])
    m_a = masses[:, 2:3]
    q = torch.sigmoid(p)
    mu = 1 - (2 * q - 1).abs()
    rho = m_a / mu.clamp_min(1e-6)
    qb, yb = (q > 0.5), (ybar > 0.5)
    inter, union = (qb & yb).sum().float(), (qb | yb).sum().float()
    return {
        'ma_mean': m_a.mean().item(),
        'rho_std': rho[mu > 0.05].std().item() if (mu > 0.05).sum() > 1 else 0.0,
        'ctrl_bce': F.binary_cross_entropy_with_logits(p, ybar).item(),
        'cell_iou': (inter / (union + 1e-6)).item(),
    }


def split_out(model_out):
    """(logits, (p, masses)) | logits  ->  logits, p, masses"""
    if isinstance(model_out, (tuple, list)):
        logits, aux = model_out
        if isinstance(aux, (tuple, list)):
            p, masses = aux
        else:  # v0 layout
            p, masses = None, aux
        return logits, p, masses
    return model_out, None, None


class CriterionV2:
    def __init__(self, base, use_ctrl, use_objective, lam_ctrl, lam_obj, k_gt, rho_f, focus_src, diag=True):
        self.base = base
        self.use_ctrl, self.use_objective = use_ctrl, use_objective
        self.lam_ctrl, self.lam_obj, self.k_gt, self.rho_f, self.focus_src = lam_ctrl, lam_obj, k_gt, rho_f, focus_src
        self.diag = diag

    def __call__(self, model_out, target):
        terms = {}
        logits, p, masses = split_out(model_out)
        loss = self.base(logits, target)
        if p is not None and self.use_ctrl:
            lc = commit_loss(p, target)
            loss = loss + self.lam_ctrl * lc
            terms['ctrl'] = lc.item()
            if self.use_objective:
                lf = focus_loss(logits, masses, target, self.k_gt, self.rho_f, self.focus_src)
                loss = loss + self.lam_obj * lf
                terms['foc'] = lf.item()
            if self.diag:
                terms.update(controller_diagnostics(p, masses, target))
        return loss, logits, terms


def build_criterion(config):
    opt = config.get('opt', {}) or {}
    base = losses.__dict__[config.get('loss', 'BCEDiceLoss')]()
    if torch.cuda.is_available():
        base = base.cuda()
    use_ctrl = config.get('use_ctrl', False) or config.get('use_fusion', False) or \
        config.get('use_context', False) or config.get('use_objective', False)
    return CriterionV2(base, use_ctrl=use_ctrl, use_objective=config.get('use_objective', False),
                       lam_ctrl=float(config.get('lam_ctrl', 0.25)), lam_obj=float(config.get('lam_obj', 0.5)),
                       k_gt=int(opt.get('k_gt', 15)), rho_f=float(opt.get('rho_f', 0.25)),
                       focus_src=str(opt.get('focus_src', 'union')), diag=bool(opt.get('diag', True)))
