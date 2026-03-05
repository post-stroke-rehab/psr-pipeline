"""
Adapter: standardized feature tensor -> GNN-ready (batch, seq_len, in_features).

Standardized input shape: (C, W, F) or (N, C, W, F)
  - C = #channels, W = #windows, F = #features per channel per window (e.g. RMS, ZC, WL).

Caller must set GNN Config.seq_len = W and Config.in_features = C*F when using
mode="windows_as_time".
"""

import logging
import torch

logger = logging.getLogger(__name__)


def feature_tensor_to_gnn_sequences(
    feature_tensor: torch.Tensor,
    *,
    mode: str = "windows_as_time",
    log_shape: bool = False,
) -> torch.Tensor:
    """
    Convert standardized feature tensor to GNN-ready (batch, seq_len, in_features).

    Args:
        feature_tensor: Shape (C, W, F) or (N, C, W, F).
            C = channels, W = windows, F = features per channel per window.
        mode: Currently only "windows_as_time" (method A) is implemented.
            Output: seq_len=W, in_features=C*F (one node per window).
        log_shape: If True, log input and output shapes.

    Returns:
        Tensor of shape (N, seq_len, in_features):
          - (1, W, C*F) if input was (C, W, F);
          - (N, W, C*F) if input was (N, C, W, F).

    Raises:
        ValueError: If feature_tensor.ndim not in (3, 4) or any dimension <= 0.

    Note:
        When using this adapter, set Config.seq_len = W and Config.in_features = C*F
        for SEMGFingerPredictor / GNN Config.
    """
    if mode != "windows_as_time":
        raise NotImplementedError(
            f'Only mode="windows_as_time" is implemented; got mode="{mode}".'
        )

    ndim = feature_tensor.dim()
    if ndim not in (3, 4):
        raise ValueError(
            f"Expected feature_tensor with 3 or 4 dimensions (C,W,F) or (N,C,W,F), "
            f"got ndim={ndim} and shape={tuple(feature_tensor.shape)}."
        )
    for i, s in enumerate(feature_tensor.shape):
        if s <= 0:
            raise ValueError(
                f"All dimensions must be positive; shape={tuple(feature_tensor.shape)}, "
                f"dim {i} is {s}."
            )

    if ndim == 3:
        # (C, W, F) -> (1, C, W, F)
        feature_tensor = feature_tensor.unsqueeze(0)
    # Now (N, C, W, F)
    n, c, w, f = feature_tensor.shape
    # (N, C, W, F) -> (N, W, C, F) -> (N, W, C*F)
    out = feature_tensor.permute(0, 2, 1, 3).reshape(n, w, -1)

    if log_shape:
        in_shape = (n, c, w, f) if ndim == 4 else (c, w, f)
        logger.info(
            "Adapter input shape: %s -> output shape: %s",
            in_shape,
            tuple(out.shape),
        )

    return out
