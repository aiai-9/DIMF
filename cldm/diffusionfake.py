# code/ImageDifussionFake/cldm/diffusionfake.py

import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import (
    log_txt_as_img,
    exists,
    default,
    ismap,
    isimage,
    mean_flat,
    count_params,
    instantiate_from_config,
)

from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.linear import Linear
from functools import partial
import copy
import pdb

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from utils.scheduler_many import build_scheduler


import timm

try:
    from timm.models import (
        vit_base_patch16_224,
        tf_efficientnet_b0_ns,
        tf_efficientnet_b4_ns,
        tf_efficientnet_b3_ns,
        tf_efficientnet_b5_ns,
        tf_efficientnet_b2_ns,
        tf_efficientnet_b6_ns,
        tf_efficientnet_b7_ns,
        xception,
    )
    from efficientnet_pytorch.model import EfficientNet
except Exception:
    from timm.models import (
        vit_base_patch16_224,
        tf_efficientnet_b0_ns,
        tf_efficientnet_b4_ns,
        tf_efficientnet_b3_ns,
        tf_efficientnet_b5_ns,
        tf_efficientnet_b2_ns,
        tf_efficientnet_b6_ns,
        tf_efficientnet_b7_ns,
        xception,
    )
    from efficientnet_pytorch.model import EfficientNet
    
from cldm.mamba_modules import MambaFakeHead  # BiMamba + IGL + temperature + gap


encoder_params = {
    # EfficientNet-B0
    "tf_efficientnet_b0_ns": {
        "model_name": "tf_efficientnet_b0_ns",
        "out_indices": [4],
        "resize_hw": (256, 256),
    },

    # EfficientNet-B4
    "tf_efficientnet_b4_ns": {
        "model_name": "tf_efficientnet_b4_ns",
        "out_indices": [4],
        "resize_hw": (380, 380),
    },

    # ConvNeXt V2 Base
    "convnextv2_base": {
        "model_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "out_indices": [3],
        "resize_hw": (256, 256),
    },

    # ConvNeXt V2 Large
    "convnextv2_large": {
        "model_name": "convnextv2_large.fcmae_ft_in22k_in1k",
        "out_indices": [3],
        "resize_hw": (256, 256),
    },
}

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class FeatureFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid(),
        )

        # spatial attention
        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels, 1, 3, padding=1), nn.Sigmoid())

        # FIXED: batch_first=True to avoid the (L, B, C) layout confusion
        self.self_attention = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)
        # fixed attention pool size — keeps attention O(49^2) regardless of input resolution
        self._attn_pool_size = 7

    def forward(self, x):
        feat = self.conv1(x)
        feat1 = nn.SiLU()(feat)

        # channel attention
        channel_weights = self.channel_attention(feat1)
        feat = feat * channel_weights

        # spatial attention
        spatial_weights = self.spatial_attention(feat1)
        feat = feat * spatial_weights

        B, C, H, W = feat.shape

        # ── CRITICAL FIX ──────────────────────────────────────────────────────
        # At 380×380 input, up_feat is 48×48 (2304 tokens).
        # Full self-attention at 2304 tokens = 4 GB of attention gradients per
        # forward/backward → gradient explosion → training collapse at ~3k steps.
        # Fix: pool to 7×7 for attention (49 tokens, same as 224×224 baseline),
        # apply attention, then upsample the delta back to original resolution
        # as a residual — preserving spatial detail while keeping gradients bounded.
        # ─────────────────────────────────────────────────────────────────────
        P = self._attn_pool_size
        feat_s  = F.adaptive_avg_pool2d(feat,  (P, P))   # (B, C, 7, 7)
        feat1_s = F.adaptive_avg_pool2d(feat1, (P, P))   # (B, C, 7, 7)

        # (B, C, 7, 7) → (B, 49, C)
        feat_s  = feat_s.flatten(2).transpose(1, 2).contiguous()
        feat1_s = feat1_s.flatten(2).transpose(1, 2).contiguous()

        # Attention in FP32 for AMP stability
        with torch.cuda.amp.autocast(enabled=False):
            q = feat1_s.float()
            k = feat_s.float()
            v = feat_s.float()
            attn_out = self.self_attention(q, k, v, need_weights=False)[0]  # (B, 49, C)

        attn_out = attn_out.to(feat.dtype)

        # (B, 49, C) → (B, C, 7, 7) → upsample to (B, C, H, W) → residual add
        attn_map = attn_out.transpose(1, 2).contiguous().view(B, C, P, P)
        if H != P or W != P:
            attn_map = F.interpolate(attn_map, size=(H, W), mode="bilinear", align_corners=False)

        feat = feat + attn_map   # residual: preserve original spatial detail

        feat = self.conv2(feat)
        return feat



# WeightNet removed — replaced by learnable contribution scalars in GuideNet.
# See self.contribution_source_logit / self.contribution_target_logit.

class GuideNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert use_spatial_transformer, "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # default hint block (will be replaced by EfficientNet in define_feature_filter)
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (not exists(num_attention_blocks)) or (nr < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        self.use_feature_filter = False
        self.ch = ch
        self.input_block_chans = input_block_chans

        # defaults for robust hint handling
        self.hint_adapter = nn.Identity()
        self.hint_resize_hw = (256, 256)
        
        # ablation attribute — set by train.py after model init
        self.ablation = None


    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    # =========================
    # FIXED define_feature_filter
    # =========================
    def define_feature_filter(self, encoder):
        if encoder not in encoder_params:
            raise ValueError(f"Unknown backbone: {encoder}")

        cfg = encoder_params[encoder]

        encoder_model = timm.create_model(
            cfg["model_name"],
            pretrained=True,
            features_only=True,
            out_indices=cfg["out_indices"],
        )

        self.encoder = encoder_model
        self.input_hint_block = encoder_model

        # robustly infer actual output channels from timm feature_info
        try:
            self.encoder_features = int(encoder_model.feature_info.channels()[-1])
        except Exception:
            # fallback: run a tiny dummy forward
            dummy_hw = cfg["resize_hw"]
            with torch.no_grad():
                dummy = torch.zeros(1, 3, dummy_hw[0], dummy_hw[1])
                feat = encoder_model(dummy)[0]
                self.encoder_features = int(feat.shape[1])

        # always project backbone output -> 1792 expected DIMF width
        if self.encoder_features != 1792:
            self.encoder_proj = nn.Conv2d(
                self.encoder_features,
                1792,
                kernel_size=1,
                bias=False,
            )
        else:
            self.encoder_proj = nn.Identity()

        self.hint_adapter = nn.Identity()
        self.hint_resize_hw = cfg["resize_hw"]

        # upsample layers
        self.upsample_conv_s1 = nn.ConvTranspose2d(1792, 1792, 4, 2, 1, bias=False)
        self.upsample_conv_s2 = nn.ConvTranspose2d(1792, 1792, 4, 2, 1, bias=False)

        self.upsample_conv_t1 = nn.ConvTranspose2d(1792, 1792, 4, 2, 1, bias=False)
        self.upsample_conv_t2 = nn.ConvTranspose2d(1792, 1792, 4, 2, 1, bias=False)

        self.feature_s = FeatureFilter(1792, 320)
        self.feature_t = FeatureFilter(1792, 320)

        self.fc = Linear(1792, 1)
        self.fc_s = Linear(1280, 1)
        self.fc_t = Linear(1280, 1)

        # Learnable source/target contribution scalars (replaces untrained WeightNet).
        # These receive gradients from diffusion reconstruction loss directly.
        # sigmoid(0.85)≈0.70, sigmoid(1.40)≈0.80 — target contributes more (matches paper finding).
        self.contribution_source_logit = nn.Parameter(torch.tensor(0.85))
        self.contribution_target_logit = nn.Parameter(torch.tensor(1.40))

        self.global_pool = AdaptiveAvgPool2d((1, 1))

        self.mamba_head = MambaFakeHead(
            d_model=1792,
            d_reduced=512,
            num_mamba_layers=2,
            d_state=16,
            pool_size=7,
            dropout=0.1,
            drop_path_rate=0.1,
            igl_margin=1.0,
            igl_weight=0.15,
            dual_drop_prob=0.15,   # FIX: was 0.30 — 30% drop starved dual branch of gradients
        )

        self.use_feature_filter = True

        print(f"[GuideNet] backbone={encoder} actual_encoder_features={self.encoder_features} resize={self.hint_resize_hw}")

    # =========================
    # Image preprocessing helper (shared by hint + source_img paths)
    # =========================
    def _preprocess_img(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize a face image from dataset space ([-1,1], BCHW) to
        ImageNet-normalized EfficientNet input space.
        Handles BCHW / BHWC / BHCW layouts robustly.
        """
        img = img.float()
        if img.ndim != 4:
            raise ValueError(f"_preprocess_img: expected 4D, got {img.ndim}D {tuple(img.shape)}")
        # Layout correction
        if img.shape[1] in (3, 4):
            pass
        elif img.shape[-1] in (3, 4):
            img = img.permute(0, 3, 1, 2).contiguous()
        elif img.shape[2] in (3, 4):
            img = img.permute(0, 2, 1, 3).contiguous()
        img = self.hint_adapter(img)
        if img.shape[-2:] != tuple(self.hint_resize_hw):
            img = F.interpolate(img, size=self.hint_resize_hw, mode="bilinear", align_corners=False)
        # [-1,1] → [0,1] → ImageNet normalise
        img = torch.clamp((img + 1.0) * 0.5, 0.0, 1.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        return (img - mean) / std

    # =========================
    # FIXED forward — True Dual-Identity DIMF
    # =========================
    def forward(self, x_source, x_target, hint, timesteps, context, source_img=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if not hasattr(self, "hint_adapter"):
            self.hint_adapter = nn.Identity()
        if not hasattr(self, "hint_resize_hw"):
            self.hint_resize_hw = (380, 380)

        # ── True Dual-Identity Feature Extraction ─────────────────────────────
        # When relation_data=True: source_img = actual SOURCE PERSON face
        #   (different identity from the hint = fake/real face)
        # When relation_data=False: source_img=None → fallback twin-projection mode
        #
        # Efficiency: if source_img is provided, concat along batch dim so both
        # images pass through EfficientNet in a SINGLE forward pass → same cost
        # as before but twice the semantic information.
        # ─────────────────────────────────────────────────────────────────────
        hint_proc = self._preprocess_img(hint)

        # ── Read ablation flags ───────────────────────────────────────────────
        abl = getattr(self, "ablation", None)
        _use_dual_up = bool(getattr(abl, "use_dual_upsample", True))
        _use_mamba   = bool(getattr(abl, "use_mamba_head",    True))
        _use_dimf    = bool(getattr(abl, "use_dimf",          True))
        _use_smc     = bool(getattr(abl, "use_global_smc",    True))
        ablation_flags = {"use_global_smc": _use_smc, "use_dimf": _use_dimf}

        # ── Feature extraction — ablation-gated dual-upsample ────────────────
        if _use_dual_up and source_img is not None and self.use_feature_filter:
            # Full dual-identity path
            src_proc = self._preprocess_img(source_img)
            combined_feat = self.input_hint_block(
                torch.cat([hint_proc, src_proc], dim=0)
            )[0]
            combined_feat = self.encoder_proj(combined_feat)
            feature, source_feature = combined_feat.chunk(2, dim=0)
            up_feat_s = self.upsample_conv_s1(source_feature)
            up_feat_s = self.upsample_conv_s2(up_feat_s)

        elif _use_dual_up:
            # Dual-upsample ON but no source_img: twin projection fallback
            feature = self.input_hint_block(hint_proc)[0]
            feature = self.encoder_proj(feature)
            up_feat_s = self.upsample_conv_s1(feature)
            up_feat_s = self.upsample_conv_s2(up_feat_s)

        else:
            # Ablation (a)/(b): NO dual upsample — bare backbone only
            feature = self.input_hint_block(hint_proc)[0]
            feature = self.encoder_proj(feature)
            up_feat_s = feature   # passthrough for interface compatibility
            up_feat_t = feature   # set here to skip target upsample below

        # Target stream (only when dual-upsample is on)
        if _use_dual_up:
            up_feat_t = self.upsample_conv_t1(feature)
            up_feat_t = self.upsample_conv_t2(up_feat_t)

        # ── FeatureFilter → guided hints ──────────────────────────────────────
        if _use_dual_up:
            guided_hint_source = self.feature_s(up_feat_s)
            guided_hint_target = self.feature_t(up_feat_t)
        else:
            # Ablation (a)/(b): no FeatureFilters, no guided hints into UNet
            guided_hint_source = None
            guided_hint_target = None

        # Learnable contribution weights (broadcast to batch)
        B = feature.shape[0]
        contribution_s = torch.sigmoid(self.contribution_source_logit).expand(B, 1)
        contribution_t = torch.sigmoid(self.contribution_target_logit).expand(B, 1)
        contribution = torch.cat((contribution_s, contribution_t), dim=1)

        # ── Classification head — ablation-gated ─────────────────────────────
        if _use_mamba:
            # Ablation (d)+(e)+(f)+Full: full MambaFakeHead
            _labels   = getattr(self, "_labels_for_mamba", None)
            _has_dual = (source_img is not None) and _use_dimf
            output, igl_loss, gap = self.mamba_head(
                feature, up_feat_s, up_feat_t,
                labels=_labels,
                has_dual_identity=_has_dual,
                ablation_flags=ablation_flags,
            )
        else:
            # Ablation (a)/(b)/(c): bare linear head — GAP + fc
            pooled = self.global_pool(feature).flatten(1)   # (B, 1792)
            output = self.fc(pooled)                         # (B, 1)
            igl_loss = feature.new_zeros(1).squeeze()
            gap      = feature.new_zeros(B)

        self._igl_loss = igl_loss
        self._gap      = gap

        source_outs = []
        target_outs = []

        h_source = x_source.type(self.dtype)
        h_target = x_target.type(self.dtype)

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint_source is not None:
                h_source = module(h_source, emb, context)
                h_target = module(h_target, emb, context)

                # --- FIX: match spatial size with UNet feature map ---
                if guided_hint_source.shape[-2:] != h_source.shape[-2:]:
                    guided_hint_source = F.interpolate(
                        guided_hint_source, size=h_source.shape[-2:], mode="bilinear", align_corners=False
                    )
                if guided_hint_target.shape[-2:] != h_target.shape[-2:]:
                    guided_hint_target = F.interpolate(
                        guided_hint_target, size=h_target.shape[-2:], mode="bilinear", align_corners=False
                    )

                h_source = h_source + guided_hint_source
                h_target = h_target + guided_hint_target


                guided_hint_source = None
                guided_hint_target = None
            else:
                h_source = module(h_source, emb, context)
                h_target = module(h_target, emb, context)

            source_outs.append(zero_conv(h_source, emb, context))
            target_outs.append(zero_conv(h_target, emb, context))

        h_source = self.middle_block(h_source, emb, context)
        h_target = self.middle_block(h_target, emb, context)

        source_outs.append(self.middle_block_out(h_source, emb, context))
        target_outs.append(self.middle_block_out(h_target, emb, context))

        return source_outs, target_outs, output, contribution


class DiffusionFake(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, label_key, target_stage_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.target_stage_key = target_stage_key
        self.label_key = label_key
        self.only_mid_control = only_mid_control

        # ── Learnable per-layer UNet control scales ───────────────────────────
        # Replaces static [1.0]*13 / [0.3]*13.
        # sigmoid(2.20)≈0.90 and sigmoid(-0.20)≈0.45 match our old defaults.
        # The model adapts scales per-layer — e.g. early coarse layers may need
        # full strength while later fine-grained layers need less.
        # ─────────────────────────────────────────────────────────────────────
        self.control_scale_logits_s = nn.Parameter(torch.full((13,), 2.20))
        self.control_scale_logits_t = nn.Parameter(torch.full((13,), -0.20))

        self.criterion  = torch.nn.BCEWithLogitsLoss()
        # FIX: lambda_cls 2.0 → 1.5 — classification was drowning the diffusion backbone.
        # diff_w 0.5 → 0.7 — restore diffusion backbone contribution to feature learning.
        # Together these keep cls loss competitive without overpowering the backbone.
        self.lambda_cls = 1.5
        self.lambda_gap = 0.1
        self.focal_gamma = 2.0
        self.focal_alpha = 0.25
        self.diff_w = 0.7            # FIX: was 0.5 — backbone was being under-trained

        # label_smoothing: read by LogitSaturationMonitor in train.py.
        # Starts at 0.10, decays to 0.05 over warmup epochs.
        # Used in _focal_bce to soften hard targets during early training.
        self.label_smoothing = 0.10
        
        # ablation attribute — set by train.py after model init
        self.ablation = None

        # ✅ buffers for EER/AUC over full val epoch
        self._val_probs  = []
        self._val_labels = []

    def _current_control_scales(self):
        s = torch.sigmoid(self.control_scale_logits_s).clamp(0.05, 0.95)
        t = torch.sigmoid(self.control_scale_logits_t).clamp(0.05, 0.95)
        return s, t

    def _focal_bce(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Returns (bce_mean, focal_mean) both in FP32.

        FIX: Apply label smoothing (self.label_smoothing, set by LogitSaturationMonitor)
        to prevent the classifier from becoming over-confident early in training.
        Smoothed labels: 1 → (1 - ε), 0 → ε.

        FIX: Focal/BCE mix changed from 0.8/0.2 → 0.6/0.4.
        The original 80% BCE weighting caused the model to treat easy examples
        (obvious fakes) almost as heavily as hard ones. 60/40 gives focal loss
        more influence so the model learns from subtle boundary cases.
        """
        eps = float(getattr(self, "label_smoothing", 0.0))
        smooth_labels = labels.float() * (1.0 - eps) + (1.0 - labels.float()) * eps

        bce   = F.binary_cross_entropy_with_logits(logits.float(), smooth_labels)
        probs = torch.sigmoid(logits.float())
        pt    = smooth_labels * probs + (1.0 - smooth_labels) * (1.0 - probs)
        alpha_t = smooth_labels * self.focal_alpha + (1.0 - smooth_labels) * (1.0 - self.focal_alpha)
        focal = alpha_t * (1.0 - pt.clamp(1e-6)).pow(self.focal_gamma) * F.binary_cross_entropy_with_logits(
            logits.float(), smooth_labels, reduction="none"
        )
        # FIX: 0.6/0.4 mix (was 0.8/0.2) — more weight on hard examples via focal
        return bce, focal.mean()

    # =========================
    # FIXED get_input (no duplicate control=, robust layout)
    # =========================
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        source, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        target, _ = super().get_input(batch, self.target_stage_key, *args, **kwargs)
        label = batch[self.label_key].float()

        # optional scores
        source_score = batch.get("source_score", None)
        target_score = batch.get("target_score", None)

        if source_score is None:
            source_score = torch.ones(label.shape[0], device=label.device, dtype=torch.float32)
        else:
            source_score = torch.as_tensor(source_score, dtype=torch.float32, device=label.device).view(-1)

        if target_score is None:
            target_score = torch.ones(label.shape[0], device=label.device, dtype=torch.float32)
        else:
            target_score = torch.as_tensor(target_score, dtype=torch.float32, device=label.device).view(-1)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]

        if not isinstance(control, torch.Tensor):
            control = torch.as_tensor(control)

        control = control.to(self.device)

        if control.ndim != 4:
            raise TypeError(
                f"control must be a 4D torch.Tensor, got {type(control)} with shape {getattr(control, 'shape', None)}"
            )

        # accept BCHW / BHWC / BHCW
        if control.shape[1] in (3, 4):
            pass
        elif control.shape[-1] in (3, 4):
            control = control.permute(0, 3, 1, 2).contiguous()
        elif control.shape[2] in (3, 4):
            control = control.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Unexpected control shape: {tuple(control.shape)}")

        control = control.to(memory_format=torch.contiguous_format).float()

        has_scores = bool(batch.get("has_similarity_scores", False))

        # ── True Dual-Identity: raw source face for DIMF ──────────────────────
        # When relation_data=True: batch["source"] is the ORIGINAL SOURCE PERSON face.
        # We need it in pixel space (NOT VAE-encoded) for EfficientNet in GuideNet.
        # batch["source"] is already a BCHW float tensor in [-1,1] (from KeyAdapter).
        # ─────────────────────────────────────────────────────────────────────
        source_img_raw = batch.get("source", None)
        if source_img_raw is not None and isinstance(source_img_raw, torch.Tensor):
            source_img_raw = source_img_raw.float()
            if bs is not None:
                source_img_raw = source_img_raw[:bs]
            source_img_raw = source_img_raw.to(self.device)
            # Ensure BCHW
            if source_img_raw.ndim == 4:
                if source_img_raw.shape[1] not in (3, 4) and source_img_raw.shape[-1] in (3, 4):
                    source_img_raw = source_img_raw.permute(0, 3, 1, 2).contiguous()
            else:
                source_img_raw = None  # unexpected shape, skip
        else:
            source_img_raw = None

        return source, target, dict(
            c_crossattn=[c],
            c_concat=[control],
            source_img=source_img_raw,   # raw source face for DIMF dual-identity
        ), (label, source_score, target_score, has_scores)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)

        is_tuple = isinstance(x_noisy, tuple)
        if is_tuple:
            x_noisy_s, x_noisy_t = x_noisy
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)

        if cond["c_concat"] is None:
            x_in = x_noisy_s if is_tuple else x_noisy
            eps = diffusion_model(x=x_in, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)

            # ✅ return a dummy logit tensor so BCE doesn't crash
            # (better: compute real logits from a classifier branch, but this prevents failure)
            B = x_in.shape[0]
            output = torch.zeros(B, 1, device=x_in.device, dtype=torch.float32)

            return eps, eps, output

        if is_tuple:
            control_source, control_target, output, contribution = self.control_model(
                x_source=x_noisy_s, x_target=x_noisy_t,
                hint=torch.cat(cond["c_concat"], 1),
                source_img=cond.get("source_img", None),
                timesteps=t, context=cond_txt
            )

            if not torch.isfinite(contribution).all():
                raise RuntimeError("Non-finite contribution detected in tuple apply_model()")

            contribution = contribution.clamp(1e-4, 1.0)
            contribution = contribution / contribution.sum(dim=1, keepdim=True).clamp_min(1e-6)

            self.source_weight = contribution[:, 0]
            self.target_weight = contribution[:, 1]

            scale_s, scale_t = self._current_control_scales()
            control_source = [c * s_scale.view(1, 1, 1, 1) for c, s_scale in zip(control_source, scale_s)]
            control_target = [c * t_scale.view(1, 1, 1, 1) for c, t_scale in zip(control_target, scale_t)]

            control_source = [c * self.source_weight.view(-1, 1, 1, 1) for c in control_source]
            control_target = [c * self.target_weight.view(-1, 1, 1, 1) for c in control_target]

            eps_source = diffusion_model(x=x_noisy_s, timesteps=t, context=cond_txt, control=control_source, only_mid_control=self.only_mid_control)
            eps_target = diffusion_model(x=x_noisy_t, timesteps=t, context=cond_txt, control=control_target, only_mid_control=self.only_mid_control)
            return eps_source, eps_target, output

        else:
            control_source, control_target, output, contribution = self.control_model(
                x_source=x_noisy, x_target=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                source_img=cond.get("source_img", None),
                timesteps=t, context=cond_txt
            )

            if not torch.isfinite(contribution).all():
                raise RuntimeError("Non-finite contribution detected in non-tuple apply_model()")

            contribution = contribution.clamp(1e-4, 1.0)
            contribution = contribution / contribution.sum(dim=1, keepdim=True).clamp_min(1e-6)

            self.source_weight = contribution[:, 0]
            self.target_weight = contribution[:, 1]

            scale_s, scale_t = self._current_control_scales()
            control_source = [c * s_scale.view(1, 1, 1, 1) for c, s_scale in zip(control_source, scale_s)]
            control_target = [c * t_scale.view(1, 1, 1, 1) for c, t_scale in zip(control_target, scale_t)]

            control_source = [c * self.source_weight.view(-1, 1, 1, 1) for c in control_source]
            control_target = [c * self.target_weight.view(-1, 1, 1, 1) for c in control_target]

            eps_source = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control_source, only_mid_control=self.only_mid_control)
            eps_target = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control_target, only_mid_control=self.only_mid_control)
            return eps_source, eps_target, output

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None
        log = dict()

        z_s, z_t, c, label = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z_s.shape[0], N)
        n_row = min(z_s.shape[0], n_row)

        log["reconstruction"] = self.decode_first_stage(z_s)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z_s[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t_ = repeat(torch.tensor([t]), "1 -> b", b=n_row).to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t_, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c]}, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def shared_step(self, batch, **kwargs):
        source, target, c, label = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(source, target, c, label)
        return loss, loss_dict

    def forward(self, source, target, c, label, *args, **kwargs):
        if self.training:
            t = torch.randint(0, self.num_timesteps, (source.shape[0],), device=self.device).long()
        else:
            t = torch.zeros(source.shape[0], device=self.device, dtype=torch.long)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        # Inject labels so GuideNet._labels_for_mamba is set before apply_model
        if self.training:
            ce_lbl = label[0] if isinstance(label, tuple) else label
            self.control_model._labels_for_mamba = ce_lbl.float().view(-1).to(self.device)
        else:
            self.control_model._labels_for_mamba = None

        loss, bceloss, loss_dict = self.p_losses((source, target), c, t, label, *args, **kwargs)

        # ── IGL loss (stored by GuideNet during apply_model) ─────────────────
        igl_loss = getattr(self.control_model, "_igl_loss", torch.zeros(1, device=self.device)).squeeze()
        if not torch.isfinite(igl_loss):
            raise RuntimeError("Non-finite igl_loss detected")

        a_reg = torch.zeros(1, device=self.device).squeeze()
        if self.training and hasattr(self.control_model, "mamba_head"):
            a_reg = self.control_model.mamba_head.a_log_reg()
            if not torch.isfinite(a_reg):
                raise RuntimeError("Non-finite a_log regularization detected")

        # ── Gap regression loss (real→gap≈0, fake→gap≈√2) ───────────────────
        gap_loss = torch.zeros(1, device=self.device).squeeze()
        if self.training:
            gap = getattr(self.control_model, "_gap", None)
            if gap is not None:
                ce_lbl_flat = (label[0] if isinstance(label, tuple) else label).float().view(-1).to(self.device)
                gap = gap.float().view(-1)

                if not torch.isfinite(gap).all():
                    raise RuntimeError("Non-finite gap detected")

                pull = (1.0 - ce_lbl_flat) * gap.pow(2)
                push = ce_lbl_flat * F.relu(1.2 - gap).pow(2)  # target gap=1.2 for fakes
                gap_loss = (pull + push).mean()

                if not torch.isfinite(gap_loss):
                    raise RuntimeError("Non-finite gap_loss detected")

        # ── Ablation loss gates ───────────────────────────────────────────────
        abl = getattr(self, "ablation", None)
        _use_diff = bool(getattr(abl, "use_diffusion_loss", True))
        _use_igl  = bool(getattr(abl, "use_igl",            True))
        _use_gap  = bool(getattr(abl, "use_gap_loss",        True))

        if not _use_igl:
            igl_loss = igl_loss * 0.0
        if not _use_gap:
            gap_loss = gap_loss * 0.0

        diff_w = float(self.diff_w) if _use_diff else 0.0
        lambda_a_reg = 1e-4
        loss = (diff_w * loss) + (self.lambda_cls * bceloss) + igl_loss + (self.lambda_gap * gap_loss) + (lambda_a_reg * a_reg)

        # ── Temperature regularization — anchor τ near 1.0 to prevent logit scale drift ──
        if self.training and hasattr(self.control_model, "mamba_head"):
            tau = self.control_model.mamba_head.temperature
            # FIX: Strengthen tau regularization 0.01 → 0.05.
            # The weak 0.01 penalty let temperature drift freely, causing the
            # EER threshold to oscillate as the logit scale shifted each epoch.
            # 0.05 anchors τ near 1.0 without over-constraining calibration.
            tau_reg = 0.05 * (tau - 1.0).pow(2)
            loss = loss + tau_reg
        else:
            tau_reg = torch.zeros(1, device=self.device)

        loss_dict["t/diff_w"]      = torch.tensor(diff_w, device=self.device)
        loss_dict["t/l_igl"]       = igl_loss.detach()
        loss_dict["t/l_gap"]       = gap_loss.detach()
        loss_dict["t/l_a_reg"]     = a_reg.detach()
        loss_dict["t/l_tau_reg"]   = tau_reg.detach() if isinstance(tau_reg, torch.Tensor) else torch.tensor(0.0)
        loss_dict["t/loss_total"]  = loss.detach()
        loss_dict["t/lambda_cls"]  = torch.tensor(self.lambda_cls, device=self.device)
        tau_val = self.control_model.mamba_head.temperature.item() if hasattr(self.control_model, "mamba_head") else 1.0
        loss_dict["t/tau"] = torch.tensor(tau_val, device=self.device)

        # Log branch weights for monitoring
        if hasattr(self.control_model, "mamba_head"):
            w = F.softmax(self.control_model.mamba_head.logit_weights.detach(), dim=0)
            loss_dict["t/w_global"]  = w[0]
            loss_dict["t/w_spatial"] = w[1]
            loss_dict["t/w_dual"]    = w[2]

        return loss, loss_dict

    def p_losses(self, x_start, cond, t, labels, noise=None):
        if isinstance(x_start, tuple):
            x_start_s = x_start[0]
            x_start_t = x_start[1]
            noise = default(noise, lambda: torch.randn_like(x_start_s))
            x_noisy_s = self.q_sample(x_start=x_start_s, t=t, noise=noise)
            x_noisy_t = self.q_sample(x_start=x_start_t, t=t, noise=noise)
            source_output, target_output, output = self.apply_model((x_noisy_s, x_noisy_t), t, cond)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            source_output, target_output, output = self.apply_model(x_noisy, t, cond)

        if isinstance(labels, tuple):
            ce_labels = labels[0].float()
            source_score = labels[1].float()
            target_score = labels[2].float()
            has_scores = bool(labels[3]) if len(labels) > 3 else False
        else:
            ce_labels = labels.float()
            source_score = torch.ones_like(ce_labels, device=self.device)
            target_score = torch.ones_like(ce_labels, device=self.device)
            has_scores = False

        cls_output = output
        loss_dict = {}
        prefix = "t" if self.training else "v"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # =========================
        # classification — focal + BCE combined
        # =========================
        logits = cls_output.view(-1).float()
        ce_labels = ce_labels.view(-1).float()

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits detected before BCE/focal")

        # clamp BEFORE BCE/focal, not after
        logits = torch.clamp(logits, -20.0, 20.0)

        with torch.cuda.amp.autocast(enabled=False):
            bce_main, focal_main = self._focal_bce(logits, ce_labels)
            # FIX: 0.6/0.4 mix (was 0.8/0.2) — more hard-example focus via focal term.
            # Original 0.8 BCE weight treated easy and hard examples nearly equally.
            bce_loss = 0.6 * bce_main + 0.4 * focal_main

        if not torch.isfinite(bce_loss):
            raise RuntimeError("Non-finite bce_loss detected")

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = (preds == ce_labels).float().mean()

        loss_dict.update({f"{prefix}/l_ce": bce_loss, f"{prefix}/acc": acc})

        # store for EER/AUC only in validation
        if not self.training:
            loss_dict.update({f"{prefix}/probs": probs.detach()})
            loss_dict.update({f"{prefix}/labels": ce_labels.detach()})

        # loss_simple_source = self.get_loss(source_output, target, mean=False).mean([1, 2, 3])
        # loss_simple_target = self.get_loss(target_output, target, mean=False).mean([1, 2, 3])
        
        with torch.cuda.amp.autocast(enabled=False):
            loss_simple_source = self.get_loss(source_output.float(), target.float(), mean=False).mean([1, 2, 3])
            loss_simple_target = self.get_loss(target_output.float(), target.float(), mean=False).mean([1, 2, 3])

        if not torch.isfinite(loss_simple_source).all():
            raise RuntimeError("Non-finite loss_simple_source detected")
        if not torch.isfinite(loss_simple_target).all():
            raise RuntimeError("Non-finite loss_simple_target detected")

        loss_dict.update({f"{prefix}/l_sour": loss_simple_source.mean()})
        loss_dict.update({f"{prefix}/l_targ": loss_simple_target.mean()})

        # logvar_t = self.logvar[t].to(self.device)
        # loss_source = loss_simple_source / torch.exp(logvar_t) + logvar_t
        # loss_target = loss_simple_target / torch.exp(logvar_t) + logvar_t
        # loss = loss_source + loss_target
        
        # --- NUMERIC SAFE logvar weighting ---
        logvar_t = self.logvar[t].to(self.device).float()
        logvar_t = torch.clamp(logvar_t, min=-6.0, max=6.0)

        if not torch.isfinite(logvar_t).all():
            raise RuntimeError("Non-finite logvar_t detected")

        inv_var = torch.exp(-logvar_t)
        loss_source = loss_simple_source.float() * inv_var + logvar_t
        loss_target = loss_simple_target.float() * inv_var + logvar_t
        loss = loss_source + loss_target

        if not torch.isfinite(loss).all():
            raise RuntimeError("Non-finite diffusion loss detected after logvar weighting")

        # WeightNet replaced with learnable scalars — no weight loss needed.
        # Log contribution values for monitoring.
        loss_dict["w_l"] = torch.tensor(0.0, device=self.device)
        loss_dict["t/contrib_s"] = self.source_weight.mean().detach() if hasattr(self, 'source_weight') else torch.tensor(0.0)
        loss_dict["t/contrib_t"] = self.target_weight.mean().detach() if hasattr(self, 'target_weight') else torch.tensor(0.0)

        if self.learn_logvar:
            loss_dict.update({f"{prefix}/l_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = loss.mean()
        loss = self.l_simple_weight * loss
        
        
        w = self.lvlb_weights[t].to(self.device).float()

        if not torch.isfinite(w).all():
            raise RuntimeError("Non-finite lvlb_weights detected")

        w = torch.clamp(w, 0.0, 10.0)

        loss_vlb_source = self.get_loss(source_output, target, mean=False).mean(dim=(1, 2, 3))
        # loss_vlb_source = (self.lvlb_weights[t] * loss_vlb_source).mean()
        loss_vlb_source = (w * loss_vlb_source.float()).mean()
        
        loss_vlb_target = self.get_loss(target_output, target, mean=False).mean(dim=(1, 2, 3))
        # loss_vlb_target = (self.lvlb_weights[t] * loss_vlb_target).mean()
        loss_vlb_target = (w * loss_vlb_target.float()).mean()
        loss_vlb = loss_vlb_target + loss_vlb_source

        loss_dict.update({f"{prefix}/l_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, bce_loss, loss_dict

    def configure_optimizers(self):
        args = getattr(self, "args", None)

        lr = float(getattr(self, "learning_rate", 3e-4))
        if args is not None and hasattr(args, "train") and hasattr(args.train, "lr"):
            lr = float(args.train.lr)

        wd = 0.0
        if args is not None and hasattr(args, "train") and hasattr(args.train, "weight_decay"):
            wd = float(args.train.weight_decay)

        head_lr_mult = 1.2
        if args is not None and hasattr(args, "train") and hasattr(args.train, "head_lr_mult"):
            head_lr_mult = float(args.train.head_lr_mult)

        head_params = []
        base_params = []

        for name, p in self.control_model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in [
                "mamba_head",
                "control_scale_logits",
                "contribution_source_logit",
                "contribution_target_logit",
            ]):
                head_params.append(p)
            else:
                base_params.append(p)

        head_params += [self.control_scale_logits_s, self.control_scale_logits_t]

        param_groups = [
            {"params": base_params, "lr": lr, "weight_decay": wd},
            {"params": head_params, "lr": lr * head_lr_mult, "weight_decay": wd},
        ]

        if not self.sd_locked:
            param_groups.append({
                "params": list(self.model.diffusion_model.output_blocks.parameters())
                        + list(self.model.diffusion_model.out.parameters()),
                "lr": lr * 0.5,
                "weight_decay": wd,
            })

        opt = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
        )

        if args is not None and hasattr(args, "train") and hasattr(args.train, "scheduler"):
            return build_scheduler(
                optimizer=opt,
                trainer=self.trainer,
                sched_cfg_any=args.train.scheduler,
                base_lr=lr,
            )

        return opt

    @staticmethod
    def _reset_mismatched_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
        """
        After loading a checkpoint, scan the optimizer state and zero-reset any
        momentum buffer whose shape does not match its parameter.  Returns the
        number of buffers that were reset (0 = no issue, log if > 0).

        Call this from on_train_start or after ckpt_path is resolved if you need
        a post-hoc safety net beyond on_load_checkpoint.
        """
        reset = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in optimizer.state:
                    continue
                pstate = optimizer.state[p]
                for buf_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    buf = pstate.get(buf_name)
                    if buf is not None and isinstance(buf, torch.Tensor):
                        if buf.shape != p.shape:
                            pstate[buf_name] = torch.zeros_like(p)
                            reset += 1
        return reset


    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    # ==============================================================================================
    # Checkpoint shape healing — fixes AdamW _multi_tensor_adam crash on resume
    # ==============================================================================================

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Heal optimizer state shape mismatches before AdamW processes them.

        ROOT CAUSE of: RuntimeError: output with shape [] doesn't match broadcast shape [1]

        When a parameter's shape changes between runs (e.g. log_temperature was
        torch.zeros(1) → shape (1,) in old ckpt, changed to torch.tensor(-0.22)
        → shape () in new code), AdamW's _multi_tensor_adam cannot broadcast the
        saved exp_avg / exp_avg_sq momentum buffers (shape (1,)) against the new
        parameter (shape ()). The crash happens at the very first optimizer.step().

        This hook runs BEFORE Lightning restores optimizer state, giving us a
        chance to reshape all momentum buffers to match the current parameter shapes.

        Fix strategy: walk every param group in the optimizer state, find any
        tensor whose shape doesn't match the current parameter shape, and reshape it.
        """
        opt_states = checkpoint.get("optimizer_states", [])
        if not opt_states:
            return

        # Build a flat map: param_id → current parameter shape
        current_shapes = {}
        try:
            param_id = 0
            for group in self.optimizers().param_groups:
                for p in group["params"]:
                    current_shapes[param_id] = p.shape
                    param_id += 1
        except Exception:
            # optimizers() not yet available — skip healing, not critical
            return

        healed = 0
        for opt_state in opt_states:
            state = opt_state.get("state", {})
            for pid, param_state in state.items():
                target_shape = current_shapes.get(int(pid))
                if target_shape is None:
                    continue
                for buf_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    buf = param_state.get(buf_name)
                    if buf is not None and isinstance(buf, torch.Tensor):
                        if buf.shape != target_shape:
                            try:
                                param_state[buf_name] = buf.reshape(target_shape)
                                healed += 1
                            except Exception:
                                # If reshape fails (incompatible sizes), zero the buffer
                                # so AdamW restarts momentum for this parameter only.
                                param_state[buf_name] = torch.zeros(
                                    target_shape,
                                    dtype=buf.dtype,
                                    device=buf.device,
                                )
                                healed += 1

        if healed > 0:
            print(f"[on_load_checkpoint] Healed {healed} optimizer buffer shape mismatches.",
                  flush=True)
            
            
    def on_train_start(self) -> None:
        """
        Final safety net: after checkpoint restore AND optimizer init, scan for
        any residual momentum buffer shape mismatches and zero-reset them.
        This catches cases where on_load_checkpoint fired before the optimizer
        was fully built (e.g. when using scheduler dicts).
        """
        try:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
            total_reset = 0
            for opt in opts:
                if hasattr(opt, "optimizer"):
                    total_reset += self._reset_mismatched_optimizer_state(opt.optimizer)
                else:
                    total_reset += self._reset_mismatched_optimizer_state(opt)
            if total_reset > 0:
                print(
                    f"[on_train_start] Reset {total_reset} mismatched optimizer "
                    f"momentum buffers. Training will proceed normally.",
                    flush=True,
                )
        except Exception as e:
            # Non-fatal — just warn, don't block training
            print(f"[on_train_start] optimizer shape-heal skipped: {e}", flush=True)

    # ==============================================================================================
    # Validation step with EER/AUC logging
    # ==============================================================================================
    
    def _dist_is_initialized(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @torch.no_grad()
    def _gather_1d_variable(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gather variable-length 1D tensors from all ranks.
        Returns concatenated 1D tensor on rank0, local tensor on other ranks.

        NCCL-SAFE: uses torch.distributed.all_gather() directly instead of
        Lightning's self.all_gather() which uses ALLREDUCE internally and
        deadlocks when ranks enter on_validation_epoch_end at different times
        (e.g. when val drop_last=False causes uneven batch counts across ranks).

        The caller MUST call torch.distributed.barrier() before this method
        to ensure all ranks have finished their val batches.
        """
        if not self._dist_is_initialized():
            return x

        import torch.distributed as dist

        x = x.contiguous().float().view(-1).to(self.device)
        world = dist.get_world_size()

        # Step 1: exchange sizes via direct all_gather (barrier-safe)
        local_n = torch.tensor([x.numel()], device=self.device, dtype=torch.long)
        all_n   = [torch.zeros(1, device=self.device, dtype=torch.long)
                   for _ in range(world)]
        dist.all_gather(all_n, local_n)
        sizes = [int(s.item()) for s in all_n]
        max_n = max(sizes)

        # Step 2: pad to uniform length
        if x.numel() < max_n:
            x = torch.cat([x, x.new_zeros(max_n - x.numel())])

        # Step 3: gather uniform-length tensors
        gathered = [x.new_zeros(max_n) for _ in range(world)]
        dist.all_gather(gathered, x)

        if self.global_rank == 0:
            return torch.cat([gathered[r][:sizes[r]] for r in range(world)])
        return x
                
    def validation_step(self, batch, batch_idx):
        source, target, c, labels = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(source, target, c, labels)

        # log scalar metrics
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(
                    k, v,
                    on_step=False, on_epoch=True,
                    prog_bar=(k in ["v/acc", "v/l_ce", "v/loss"]),
                    sync_dist=True
                )

        # collect probs/labels for EER/AUC — store on CPU to prevent VRAM accumulation
        # across 400+ val batches (52500 samples / 32 bs = 411 batches/rank)
        if "v/probs" in loss_dict and "v/labels" in loss_dict:
            self._val_probs.append(loss_dict["v/probs"].detach().float().cpu())
            self._val_labels.append(loss_dict["v/labels"].detach().float().cpu())

        return loss

    def on_validation_epoch_start(self):
        # reset buffers cleanly every epoch
        self._val_probs = []
        self._val_labels = []


    def on_validation_epoch_end(self):
        if len(self._val_probs) == 0:
            return

        # ---- NCCL FIX: barrier before ANY collective ----
        # Ensures ALL ranks have finished their validation batches before
        # attempting cross-rank communication. Without this, rank 0 can
        # enter _gather_1d_variable while other ranks are still running
        # val batches, causing NCCL ALLREDUCE to time out after 30+ minutes.
        if self._dist_is_initialized():
            torch.distributed.barrier()

        # ---- concat local buffers — keep on CPU to save VRAM ----
        probs_local = torch.cat(self._val_probs, dim=0).float().view(-1).to(self.device)
        labs_local  = torch.cat(self._val_labels, dim=0).float().view(-1).to(self.device)

        # clear buffers early (avoid growth)
        self._val_probs.clear()
        self._val_labels.clear()

        # ---- gather across ranks (variable-length safe) ----
        probs_all = self._gather_1d_variable(probs_local)
        labs_all  = self._gather_1d_variable(labs_local)

        # barrier after gather, before broadcast
        if self._dist_is_initialized():
            torch.distributed.barrier()

        # ---- compute only on rank0 ----
        if self.global_rank == 0:
            probs_np = probs_all.detach().cpu().numpy().astype(np.float64)
            labs_np  = labs_all.detach().cpu().numpy().astype(np.float64)

            # sanitize
            mask = np.isfinite(probs_np) & np.isfinite(labs_np)
            probs_np = probs_np[mask]
            labs_np  = labs_np[mask]

            # labs should be {0,1}
            labs_np = (labs_np > 0.5).astype(np.int32)

            # defaults
            eer = 0.5
            auc = 0.5
            best_acc = 0.0
            acc_at_eer = 0.0
            eer_thr = 0.5

            if probs_np.size > 0 and len(np.unique(labs_np)) >= 2:
                # roc curve
                fpr, tpr, thr = roc_curve(labs_np, probs_np, pos_label=1)
                fnr = 1.0 - tpr

                # EER index (closest point where FPR ~ FNR)
                idx = int(np.nanargmin(np.abs(fpr - fnr)))
                eer = float(fpr[idx])          # standard EER estimate
                eer_thr = float(thr[idx])

                # AUC
                auc = float(roc_auc_score(labs_np, probs_np))

                # Best accuracy across thresholds (not fixed 0.5)
                # thr returned by roc_curve is length = len(fpr) = len(tpr)
                accs = []
                for th_ in thr:
                    pred = (probs_np >= th_).astype(np.int32)
                    accs.append((pred == labs_np).mean())
                best_acc = float(np.max(accs)) if len(accs) > 0 else 0.0

                # Accuracy at EER threshold
                pred_eer = (probs_np >= eer_thr).astype(np.int32)
                acc_at_eer = float((pred_eer == labs_np).mean())

            # pack tensors on device
            eer_t = torch.tensor(eer, device=self.device, dtype=torch.float32)
            auc_t = torch.tensor(auc, device=self.device, dtype=torch.float32)
            best_acc_t = torch.tensor(best_acc, device=self.device, dtype=torch.float32)
            acc_at_eer_t = torch.tensor(acc_at_eer, device=self.device, dtype=torch.float32)
            eer_thr_t = torch.tensor(eer_thr, device=self.device, dtype=torch.float32)

        else:
            eer_t = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            auc_t = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            best_acc_t = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            acc_at_eer_t = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            eer_thr_t = torch.tensor(0.5, device=self.device, dtype=torch.float32)

        # ---- broadcast rank0 results to all ranks ----
        if self._dist_is_initialized():
            torch.distributed.broadcast(eer_t, src=0)
            torch.distributed.broadcast(auc_t, src=0)
            torch.distributed.broadcast(best_acc_t, src=0)
            torch.distributed.broadcast(acc_at_eer_t, src=0)
            torch.distributed.broadcast(eer_thr_t, src=0)

        # ---- log ----
        self.log("v/eer", eer_t, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("v/auc", auc_t, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("v/best_acc", best_acc_t, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("v/acc_at_eer", acc_at_eer_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("v/eer_thr", eer_thr_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)