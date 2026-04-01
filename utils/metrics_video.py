# utils/metrics_video.py  [FIXED v2]
# =============================================================
# FIXES vs original:
#
# FIX-M1: path_to_video_id — correct fallback for different dataset layouts
#   Original: os.path.dirname(path) as video_id
#   Bug: for flat-file datasets (DiffSwap, WildDeepfake), all reals share
#   the same parent dir -> N_videos=2, V-AUC=1.0 (completely artificial).
#   Fix: use the video_id field from the dataset batch dict (done in
#   eval_generalization.py via compute_video_metrics_with_ids).
#   This file keeps the original path_to_video_id as a fallback only.
#
# FIX-M2: compute_metrics EER formula
#   Use (fpr[idx] + fnr[idx]) / 2 for the EER estimate — more accurate
#   than fpr[idx] alone, especially when the crossing isn't exact.
#   (This was already correct in the version you shared — kept as-is.)
#
# FIX-M3: compute_video_metrics — accept explicit video_id list
#   Added video_ids parameter (optional). When provided, uses those IDs
#   for grouping instead of path_to_video_id. Backward compatible.
# =============================================================

import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_metrics(probs_np, labs_np):
    probs_np = np.asarray(probs_np, dtype=np.float64).reshape(-1)
    labs_np  = np.asarray(labs_np,  dtype=np.float64).reshape(-1)
    labs_np  = (labs_np > 0.5).astype(np.int32)

    if probs_np.shape[0] != labs_np.shape[0]:
        raise ValueError(
            f"compute_metrics mismatch: "
            f"len(probs)={len(probs_np)} len(labels)={len(labs_np)}"
        )

    mask     = np.isfinite(probs_np) & np.isfinite(labs_np)
    probs_np = probs_np[mask]
    labs_np  = labs_np[mask]

    eer       = 0.5
    auc       = 0.5
    auc_flip  = 0.5
    best_acc  = 0.0
    acc_eer   = 0.0
    eer_thr   = 0.5

    if probs_np.size > 0 and len(np.unique(labs_np)) >= 2:
        fpr, tpr, thr = roc_curve(labs_np, probs_np, pos_label=1)
        fnr = 1.0 - tpr

        # FIX-M2: correct EER = midpoint of FPR and FNR at crossing
        idx     = int(np.nanargmin(np.abs(fpr - fnr)))
        eer     = float((fpr[idx] + fnr[idx]) / 2.0)
        eer_thr = float(thr[idx])

        auc      = float(roc_auc_score(labs_np, probs_np))
        auc_flip = float(roc_auc_score(labs_np, 1.0 - probs_np))

        accs     = [(probs_np >= th).astype(np.int32) == labs_np
                    for th in thr]
        accs     = [float(a.mean()) for a in accs]
        best_acc = float(np.max(accs)) if accs else 0.0

        pred_eer = (probs_np >= eer_thr).astype(np.int32)
        acc_eer  = float((pred_eer == labs_np).mean())

    return {
        "N":          int(probs_np.size),
        "EER":        eer,
        "AUC":        auc,
        "AUC_FLIPPED": auc_flip,
        "BEST_ACC":   best_acc,
        "ACC_AT_EER": acc_eer,
        "EER_THR":    eer_thr,
    }


def path_to_video_id(path: str) -> str:
    """
    Fallback: infer video_id from the file path.
    Used only when the dataset does not provide an explicit video_id.

    Correct for:
      - FF++:  .../images/000_003/frame.png  -> .../images/000_003
      - DFD:   .../DFD_orig-mtcnn/vid/frame.png -> .../vid
      - CelebDF: .../celeb-real-mtcnn/id00001/frame.png -> .../id00001

    Wrong for (use explicit video_id instead):
      - DiffSwap: .../DiffSwap/DiffSwap/2421.png -> all fakes same parent
      - WildDeepfake: .../test/real/0_102.png    -> all reals same parent
    """
    path = str(path).replace("\\", "/").strip()
    if not path:
        return "unknown_video"
    parent = os.path.dirname(path)
    if parent:
        return parent
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem if stem else path


def compute_video_metrics(probs_np, labs_np, paths,
                           top_k: int = 10, mode: str = "topk",
                           video_ids: Optional[List[str]] = None):
    """
    Compute video-level metrics by aggregating frame predictions per video.

    FIX-M3: Accepts explicit video_ids list (preferred). Falls back to
    path_to_video_id(path) when video_ids is None.

    Args:
        probs_np:   (N,) frame-level predicted probabilities
        labs_np:    (N,) frame-level binary labels
        paths:      (N,) frame file paths
        top_k:      number of top-scoring frames to average (topk mode)
        mode:       aggregation mode: 'topk' | 'mean' | 'median'
        video_ids:  (N,) explicit video IDs from dataset (FIX-M3)

    Returns:
        (metrics_dict, video_probs, video_labs, video_ids_list)
    """
    probs_np = np.asarray(probs_np, dtype=np.float64).reshape(-1)
    labs_np  = (np.asarray(labs_np).reshape(-1) > 0.5).astype(np.int32)

    if probs_np.shape[0] != labs_np.shape[0]:
        raise ValueError(
            f"compute_video_metrics mismatch: "
            f"len(probs)={len(probs_np)} len(labels)={len(labs_np)}"
        )
    if len(paths) != len(probs_np):
        raise ValueError(
            f"compute_video_metrics path mismatch: "
            f"len(paths)={len(paths)} len(probs)={len(probs_np)}"
        )

    # FIX-M3: choose video grouping key
    if video_ids is not None and len(video_ids) == len(probs_np):
        vid_keys = [str(v) if v else path_to_video_id(p)
                    for v, p in zip(video_ids, paths)]
    else:
        vid_keys = [path_to_video_id(p) for p in paths]

    video_scores = defaultdict(list)
    video_labels = {}
    for prob, lab, vid in zip(probs_np, labs_np, vid_keys):
        video_scores[vid].append(float(prob))
        video_labels[vid] = int(lab)

    v_probs, v_labs, v_ids = [], [], []
    for vid, scores in video_scores.items():
        scores = np.asarray(scores, dtype=np.float64)
        if mode == "median":
            score = float(np.median(scores))
        elif mode == "mean":
            score = float(np.mean(scores))
        else:  # topk
            k     = min(int(top_k), len(scores))
            score = float(np.mean(np.sort(scores)[-k:]))
        v_probs.append(score)
        v_labs.append(video_labels[vid])
        v_ids.append(vid)

    v_probs = np.asarray(v_probs, dtype=np.float64)
    v_labs  = np.asarray(v_labs,  dtype=np.int32)
    m       = compute_metrics(v_probs, v_labs)

    return m, v_probs, v_labs, v_ids