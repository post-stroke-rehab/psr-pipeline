from __future__ import annotations

import torch

from training.rp5_four_channel import (
    CHANNEL_POLICIES,
    CNNMicroSequence,
    FEATURES_PER_CHANNEL,
    FourChannelRunConfig,
    feature_indices_for_channels,
    load_cnn_micro_transfer,
    run_four_channel_experiment,
    stride_samples,
    window_samples,
)


def test_cnn_micro_accepts_four_channel_feature_contract():
    model = CNNMicroSequence(in_features=48, out_dim=5)
    x = torch.randn(3, 4, 48)
    y = model(x)
    assert tuple(y.shape) == (3, 5)


def test_cnn_micro_accepts_single_window_context_with_padding():
    model = CNNMicroSequence(in_features=48, out_dim=5)
    x = torch.randn(2, 1, 48)
    y = model(x)
    assert tuple(y.shape) == (2, 5)


def test_feature_indices_preserve_canonical_channel_order():
    indices = feature_indices_for_channels(CHANNEL_POLICIES["left"])
    assert len(indices) == 4 * FEATURES_PER_CHANNEL
    assert indices[:12] == list(range(0, 12))
    assert indices[12:24] == list(range(24, 36))
    assert indices[24:36] == list(range(96, 108))
    assert indices[36:48] == list(range(156, 168))


def test_window_math_matches_physiomio_contract():
    assert window_samples() == 410
    assert stride_samples() == 205


def test_transfer_slices_full64_first_layer(tmp_path):
    source = CNNMicroSequence(in_features=64 * 12, out_dim=5)
    target = CNNMicroSequence(in_features=48, out_dim=5)
    ckpt_path = tmp_path / "full64_micro.pt"
    torch.save({"model_state": source.state_dict()}, ckpt_path)

    info = load_cnn_micro_transfer(
        target,
        ckpt_path,
        source_channel_indices=CHANNEL_POLICIES["right"],
    )

    assert len(info["selected_feature_indices"]) == 48
    assert tuple(target.backbone.proj.weight.shape) == (48, 48, 1)


def test_synthetic_smoke_run_writes_checkpoint(tmp_path):
    cfg = FourChannelRunConfig(
        run_id="pytest_smoke",
        output_root=str(tmp_path),
        synthetic_smoke=True,
        epochs=1,
        batch_size=8,
        context_windows=4,
        device="cpu",
        max_train_batches=1,
        max_eval_batches=1,
    )
    result = run_four_channel_experiment(cfg)
    assert (tmp_path / "pytest_smoke" / "checkpoint_best.pt").exists()
    assert result["thresholds"]
    assert result["test_metrics"]["finger_accuracy"] >= 0.0
