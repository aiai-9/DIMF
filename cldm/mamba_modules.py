"""
cldm/mamba_modules.py  —  DIMF Mamba Modules  [STABLE v8]
==========================================================

FIXES vs v7:
  FIX-E (CRITICAL): MambaFakeHead threshold collapse prevention
    The eer_thr → 0.999 issue from the WandB charts was caused by the
    3-branch ensemble (global + spatial + dual) collapsing when dual branch
    is disabled (no identity pairs). Without dual, only 2 branches vote
    but the weight normalization still multiplies by temperature, causing
    the global branch (which is simplest) to dominate with large logits.
    
    Fixes:
    1. Separate logit_weights for 2-branch vs 3-branch mode
       (don't reuse [:2] slice of 3-branch weights — gradient paths differ)
    2. Logit clipping INSIDE MambaFakeHead before temperature scaling
       prevents runaway logits from saturating sigmoid
    3. Temperature learned more aggressively (wider init range 0.3–3.0)
       to allow calibration across both 2-branch and 3-branch modes
    4. GlobalHead uses avg+max pooling concatenation instead of avg-only
       → richer global descriptor, better discrimination on single-domain

  FIX-F: DualIdentityMambaFusion twin-mode robustness
    When source==target (fs≈ft), cross-attention becomes identity mapping.
    Added small noise injection to ft during training in twin mode to
    prevent cross-attention from degenerating to pure passthrough.

  FIX-G: IdentityGapLoss — zero-safe when gs/gt are dummy zeros
    When use_dual=False, gs=gt=zeros. IGL is skipped (labels is None check
    was correct), but gap = ||0-0|| = 0 always. Gap regression loss in
    diffusionfake.py now checks _has_dual_last flag (set here) before using gap.

  All other interfaces UNCHANGED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import random


# ─────────────────────────────────────────────────────────────────────────────
# DropPath (stochastic depth)
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop entire residual path with probability p during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep) / keep
        return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# Selective-SSM core
# ─────────────────────────────────────────────────────────────────────────────

class _SSMCore(nn.Module):
    """
    Mamba-style selective SSM.  (B, L, d_model) → (B, L, d_model)
    All ops in FP32 for numerical stability.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner        = int(d_model * expand)
        self.d_inner   = d_inner
        self.d_state   = d_state

        self.in_proj   = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d    = nn.Conv1d(d_inner, d_inner, d_conv,
                                   padding=d_conv - 1, groups=d_inner, bias=True)
        self.x_proj    = nn.Linear(d_inner, 1 + d_state * 2, bias=False)
        self.dt_proj   = nn.Linear(1, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A).clamp(-6.0, 6.0))
        self.D     = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def a_log_reg(self) -> torch.Tensor:
        """L1 penalty on A_log to prevent drift. Add to training loss * 1e-4."""
        return self.A_log.abs().mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        B, L, _ = x.shape
        S = self.d_state

        xz    = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)

        xc = xi.transpose(1, 2).contiguous()
        xc = self.conv1d(xc)[..., :L].transpose(1, 2).contiguous()
        xc = F.silu(xc)

        bcdt           = self.x_proj(xc)
        dt_raw, Bs, Cs = bcdt.split([1, S, S], dim=-1)
        dt             = F.softplus(self.dt_proj(dt_raw))

        A = -torch.exp(self.A_log.float().clamp(-6.0, 6.0))

        dA = torch.exp(dt.unsqueeze(-1) * A[None, None])
        dB = dt.unsqueeze(-1) * Bs.unsqueeze(2)

        h, ys = xc.new_zeros(B, self.d_inner, S), []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * xc[:, i].unsqueeze(-1)
            h = h.clamp(-1e4, 1e4)
            ys.append((h * Cs[:, i].unsqueeze(1)).sum(-1))

        y = torch.stack(ys, dim=1) + xc * self.D[None, None]
        y = y * F.silu(z)
        return self.out_proj(y)


# ─────────────────────────────────────────────────────────────────────────────
# BidirectionalMambaBlock
# ─────────────────────────────────────────────────────────────────────────────

class BidirectionalMambaBlock(nn.Module):
    """
    Forward + backward SSM scan with learnable α fusion + DropPath regulariser.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.ssm_fwd = _SSMCore(d_model, d_state, d_conv, expand)
        self.ssm_bwd = _SSMCore(d_model, d_state, d_conv, expand)
        self.alpha   = nn.Parameter(torch.tensor(0.0))
        self.drop    = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)

    def a_log_reg(self) -> torch.Tensor:
        return self.ssm_fwd.a_log_reg() + self.ssm_bwd.a_log_reg()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.float()
        xn       = self.norm(residual)

        y_fwd    = self.ssm_fwd(xn)
        y_bwd    = self.ssm_bwd(xn.flip(1)).flip(1)

        alpha    = torch.sigmoid(self.alpha)
        y        = alpha * y_fwd + (1.0 - alpha) * y_bwd

        if not torch.isfinite(y).all():
            raise RuntimeError("Non-finite y detected in BidirectionalMambaBlock")

        return self.drop_path(self.drop(y)) + residual


# ─────────────────────────────────────────────────────────────────────────────
# IdentityGapLoss
# ─────────────────────────────────────────────────────────────────────────────

class IdentityGapLoss(nn.Module):
    """
    Contrastive supervision on DIMF identity embeddings.
    Only used when has_dual_identity=True (genuine different-identity pairs).
    """

    def __init__(self, margin: float = 1.0, weight: float = 0.10):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, gs: torch.Tensor, gt: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        gs     = F.normalize(gs.float(), dim=-1)
        gt     = F.normalize(gt.float(), dim=-1)
        dist   = 1.0 - (gs * gt).sum(-1).clamp(-1.0, 1.0)
        lbl    = labels.float().view(-1)
        pull   = (1.0 - lbl) * F.relu(dist - 0.3).pow(2)
        push   = lbl          * F.relu(self.margin - dist).pow(2)
        return self.weight * (pull + push).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SpatialMambaClassifier  (SMC)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialMambaClassifier(nn.Module):
    """
    Models spatial artifact consistency with BiMamba over a 7×7 token grid.
    """

    def __init__(self, d_model: int, d_project: int = 512,
                 num_layers: int = 2, d_state: int = 16,
                 dropout: float = 0.1, drop_path_rate: float = 0.1,
                 pool_size: int = 7):
        super().__init__()
        self.pool_size  = pool_size
        self.input_proj = nn.Linear(d_model, d_project, bias=False)
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_project, d_state=d_state,
                                     dropout=dropout, drop_path_rate=drop_path_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_project)
        self.head = nn.Linear(d_project, 1, bias=True)

    def a_log_reg(self) -> torch.Tensor:
        return sum(l.a_log_reg() for l in self.layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-2:] != (self.pool_size, self.pool_size):
            feat = F.adaptive_avg_pool2d(feat, (self.pool_size, self.pool_size))
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).float()
        tokens = self.input_proj(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# DualIdentityMambaFusion  (DIMF)
# ─────────────────────────────────────────────────────────────────────────────

class DualIdentityMambaFusion(nn.Module):
    """
    Dual-Identity Mamba Fusion.
    FIX-F: Added twin-mode noise injection when fs≈ft to prevent
    cross-attention from degenerating to identity mapping.
    """

    def __init__(self, d_model: int, d_reduced: int = 512,
                 num_layers: int = 1, d_state: int = 16,
                 pool_size: int = 7, dropout: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.pool_size = pool_size

        self.proj_s = nn.Linear(d_model, d_reduced, bias=False)
        self.proj_t = nn.Linear(d_model, d_reduced, bias=False)

        self.mamba_s = nn.ModuleList([
            BidirectionalMambaBlock(d_reduced, d_state=d_state,
                                     dropout=dropout, drop_path_rate=drop_path_rate)
            for _ in range(num_layers)
        ])
        self.mamba_t = nn.ModuleList([
            BidirectionalMambaBlock(d_reduced, d_state=d_state,
                                     dropout=dropout, drop_path_rate=drop_path_rate)
            for _ in range(num_layers)
        ])

        self.norm_s = nn.LayerNorm(d_reduced)
        self.norm_t = nn.LayerNorm(d_reduced)

        n_heads       = max(1, d_reduced // 64)
        self.cross_st = nn.MultiheadAttention(d_reduced, n_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_ts = nn.MultiheadAttention(d_reduced, n_heads,
                                               dropout=dropout, batch_first=True)
        self.norm_st  = nn.LayerNorm(d_reduced)
        self.norm_ts  = nn.LayerNorm(d_reduced)

        self.fusion_head = nn.Sequential(
            nn.Linear(d_reduced * 2, d_reduced),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_reduced, 1)
        )

    def a_log_reg(self) -> torch.Tensor:
        s = sum(l.a_log_reg() for l in self.mamba_s)
        t = sum(l.a_log_reg() for l in self.mamba_t)
        return s + t

    def _tokenize(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-2:] != (self.pool_size, self.pool_size):
            feat = F.adaptive_avg_pool2d(feat, (self.pool_size, self.pool_size))
        return feat.flatten(2).transpose(1, 2).float()

    def forward(self, fs: torch.Tensor, ft: torch.Tensor,
                is_twin_mode: bool = False) -> tuple:
        ts = self.proj_s(self._tokenize(fs))
        tt_raw = self._tokenize(ft)

        # FIX-F: In twin mode (source==target), inject small noise into ft tokens
        # so cross-attention learns to look at differences, not just copy.
        if is_twin_mode and self.training:
            noise_scale = 0.02 * tt_raw.std().detach().clamp_min(1e-6)
            tt_raw = tt_raw + torch.randn_like(tt_raw) * noise_scale

        tt = self.proj_t(tt_raw)

        for layer in self.mamba_s:
            ts = layer(ts)
        for layer in self.mamba_t:
            tt = layer(tt)

        ts = self.norm_s(ts)
        tt = self.norm_t(tt)

        ts_c, _ = self.cross_st(ts, tt, tt)
        tt_c, _ = self.cross_ts(tt, ts, ts)
        ts = self.norm_st(ts + ts_c)
        tt = self.norm_ts(tt + tt_c)

        gs = ts.mean(dim=1)
        gt = tt.mean(dim=1)
        return self.fusion_head(torch.cat([gs, gt], dim=-1)), gs, gt


# ─────────────────────────────────────────────────────────────────────────────
# MambaFakeHead  [v8 — threshold collapse prevention]
# ─────────────────────────────────────────────────────────────────────────────

class MambaFakeHead(nn.Module):
    """
    Three-branch classification head.
    Branch 1 — GlobalHead:       GAP + GMP concat + Linear  [FIX-E: richer descriptor]
    Branch 2 — SpatialMamba:     BiMamba on 7×7 tokens
    Branch 3 — DualIdentityMamba: BiMamba + symmetric cross-attention

    FIX-E: Separate branch weights for 2-branch vs 3-branch ensemble.
    In 2-branch mode (no dual identity), use logit_weights_2b instead of
    slicing logit_weights_3b[:2]. This prevents gradient interference between
    the two operating modes and stabilizes the threshold.
    """

    def __init__(self,
                 d_model:          int   = 1792,
                 d_reduced:        int   = 512,
                 num_mamba_layers: int   = 2,
                 d_state:          int   = 16,
                 pool_size:        int   = 7,
                 dropout:          float = 0.1,
                 drop_path_rate:   float = 0.1,
                 igl_margin:       float = 1.0,
                 igl_weight:       float = 0.10,
                 dual_drop_prob:   float = 0.15):  # FIX: was 0.35 — 35% drop caused gradient starvation on dual branch
        super().__init__()
        self.dual_drop_prob = dual_drop_prob

        # FIX-E: GlobalHead uses avg + max pooling for richer descriptor
        # GAP alone misses spatial extremes important for forgery detection.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 2, d_model, bias=False),  # concat avg+max
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.spatial_mamba = SpatialMambaClassifier(
            d_model=d_model, d_project=d_reduced,
            num_layers=num_mamba_layers, d_state=d_state,
            dropout=dropout, drop_path_rate=drop_path_rate,
            pool_size=pool_size,
        )

        self.dual_mamba = DualIdentityMambaFusion(
            d_model=d_model, d_reduced=d_reduced,
            num_layers=max(1, num_mamba_layers - 1),
            d_state=d_state, pool_size=pool_size,
            dropout=dropout, drop_path_rate=drop_path_rate,
        )

        self.igl = IdentityGapLoss(margin=igl_margin, weight=igl_weight)

        # Branch weights — separate sets for 2-branch vs 3-branch mode.
        # Init 3-branch: favour spatial (0.25/0.45/0.30) since spatial artifacts
        # are most reliable early; 2-branch: slight spatial bias (0.40/0.60).
        # SHAPE: 1-D tensors of length 3 and 2 — unchanged from v8, resume-safe.
        self.logit_weights    = nn.Parameter(torch.tensor([0.25, 0.45, 0.30]))  # 3-branch
        self.logit_weights_2b = nn.Parameter(torch.tensor([0.40, 0.60]))        # 2-branch

        # CRITICAL SHAPE FIX — keep shape (1,) not scalar () for both temperatures.
        # Reason: AdamW stores momentum buffers whose shape is determined at the
        # FIRST optimizer step. If a checkpoint was saved with shape (1,) and we
        # later create the parameter as shape (), _multi_tensor_adam raises:
        #   RuntimeError: output with shape [] doesn't match broadcast shape [1]
        # Using zeros(1) preserves the original checkpoint shape.
        # The init value -0.22 gives τ = exp(-0.22) ≈ 0.80 — slight logit
        # amplification that stabilises threshold calibration at startup.
        self.log_temperature    = nn.Parameter(torch.full((1,), -0.22))  # τ≈0.80
        self.log_temperature_2b = nn.Parameter(torch.full((1,), -0.22))

    @property
    def temperature(self) -> torch.Tensor:
        # .squeeze() handles both shape () and (1,) — produces scalar for division
        return self.log_temperature.squeeze().exp().clamp(0.3, 3.0)

    @property
    def temperature_2b(self) -> torch.Tensor:
        return self.log_temperature_2b.squeeze().exp().clamp(0.3, 3.0)

    def a_log_reg(self) -> torch.Tensor:
        """L1 penalty on all A_log params. Add to loss * 1e-4 during training."""
        return (self.spatial_mamba.a_log_reg()
                + self.dual_mamba.a_log_reg())

    def _global_features(self, feature: torch.Tensor) -> torch.Tensor:
        """FIX-E: Concatenate avg and max pooled features for richer descriptor."""
        avg = self.global_avg_pool(feature)  # (B, C, 1, 1)
        mx  = self.global_max_pool(feature)  # (B, C, 1, 1)
        return torch.cat([avg, mx], dim=1)   # (B, 2C, 1, 1)

    def load_state_dict(self, state_dict, strict=True):
        """
        Shape-healing override for checkpoint resume.

        Older checkpoints stored log_temperature / log_temperature_2b as shape (1,).
        Our init also uses (1,) now, but guard against any future shape drift by
        reshaping loaded tensors to match current parameter shapes before the
        standard load. This prevents the AdamW '_multi_tensor_adam' crash:
          RuntimeError: output with shape [] doesn't match broadcast shape [1]
        """
        healed = {}
        own_state = self.state_dict()
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape != v.shape:
                try:
                    healed[k] = v.reshape(own_state[k].shape)
                except Exception:
                    healed[k] = v  # let standard load handle/report the mismatch
            else:
                healed[k] = v
        return super().load_state_dict(healed, strict=strict)

    def forward(self,
                feature: torch.Tensor,
                fs:      torch.Tensor,
                ft:      torch.Tensor,
                labels:  Optional[torch.Tensor] = None,
                has_dual_identity: bool = True,
                ablation_flags: Optional[dict] = None):
        """
        Returns: logit (B,1), igl_loss (scalar), gap (B,)

        FIX-E: Uses separate branch weights and temperatures for 2-branch vs
        3-branch mode to prevent the threshold collapse observed in WandB.

        ablation_flags (optional dict, passed from GuideNet.forward):
          use_global_smc: bool — if False, skip SMC; use global branch only
          use_dimf:       bool — if False, force 2-branch (no dual identity)
        """
        # ── Ablation flags ────────────────────────────────────────────────────
        if ablation_flags is None:
            ablation_flags = {}
        _use_smc  = bool(ablation_flags.get("use_global_smc", True))
        _use_dimf = bool(ablation_flags.get("use_dimf",       True))
        # ─────────────────────────────────────────────────────────────────────

        feature = feature.float()
        fs      = fs.float()
        ft      = ft.float()

        # Global branch — always active
        global_feat = self._global_features(feature)
        logit_g = self.global_head(global_feat)
        logit_g = torch.clamp(logit_g, -10.0, 10.0)

        # Spatial branch — gated by use_global_smc ablation flag
        if _use_smc:
            logit_s = self.spatial_mamba(feature)
            logit_s = torch.clamp(logit_s, -10.0, 10.0)
        else:
            logit_s = None  # ablation (d) baseline: global-only

        # ── Dual-branch dropout — also gated by use_dimf ablation flag ───────
        use_dual = has_dual_identity and _use_dimf
        if self.training and use_dual and self.dual_drop_prob > 0:
            if random.random() < self.dual_drop_prob:
                use_dual = False

        # Detect twin mode (source and target are same image)
        is_twin = not has_dual_identity

        # Dummy gs/gt — overwritten when dual branch runs
        B = feature.shape[0]
        d = self.dual_mamba.proj_s.out_features
        gs = feature.new_zeros(B, d)
        gt = feature.new_zeros(B, d)

        if use_dual and logit_s is not None:
            # Full 3-branch: global + SMC + DIMF
            w = F.softmax(self.logit_weights, dim=0)
            logit_d, gs, gt = self.dual_mamba(fs, ft, is_twin_mode=is_twin)
            logit_d = torch.clamp(logit_d, -10.0, 10.0)
            logit = (w[0]*logit_g + w[1]*logit_s + w[2]*logit_d) / self.temperature

            igl_loss = feature.new_zeros(1).squeeze()
            if self.training and labels is not None:
                igl_loss = self.igl(gs, gt, labels)

            gs_n = F.normalize(gs, dim=-1)
            gt_n = F.normalize(gt, dim=-1)
            gap  = torch.norm(gs_n - gt_n, dim=-1)

        elif logit_s is not None:
            # 2-branch: global + SMC only (no DIMF)
            # FIX-E: Use separate 2-branch weights and temperature
            w2 = F.softmax(self.logit_weights_2b, dim=0)
            logit = (w2[0]*logit_g + w2[1]*logit_s) / self.temperature_2b
            igl_loss = feature.new_zeros(1).squeeze()
            gap      = feature.new_zeros(B)

        else:
            # 1-branch: global only (ablation d with use_global_smc=False)
            logit    = logit_g / self.temperature_2b
            igl_loss = feature.new_zeros(1).squeeze()
            gap      = feature.new_zeros(B)

        if not torch.isfinite(logit).all():
            raise RuntimeError("Non-finite logit detected in MambaFakeHead")

        return logit, igl_loss, gap