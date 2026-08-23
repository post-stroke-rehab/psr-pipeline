from pathlib import Path

import torch
import torch.nn.functional as F

from models.CNN.export_onnx import (
    AdaptiveCNNStudent,
    CNNMicroSequence,
    StaticAdaptiveMaxPool1d,
    load_checkpoint_model,
    model_input_features,
    model_output_features,
    prepare_export_model,
    resolve_windows,
)


def test_loads_four_channel_cnn_micro_sequence_checkpoint(tmp_path: Path) -> None:
    source = CNNMicroSequence(in_features=48, out_dim=5, dropout=0.2)
    checkpoint = tmp_path / "four_channel.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "config": {"dropout": 0.2, "context_windows": 9},
            "model_config": {
                "architecture": "CNN_Micro",
                "input_shape": ["batch", "windows", 48],
                "output_shape": ["batch", 5],
            },
        },
        checkpoint,
    )

    model, payload = load_checkpoint_model(checkpoint)

    assert isinstance(model, CNNMicroSequence)
    assert model_input_features(model) == 48
    assert model_output_features(model) == 5
    assert resolve_windows(payload, None) == 9

    x = torch.randn(2, 9, 48)
    with torch.inference_mode():
        expected = source.eval()(x)
        actual = model(x)
    torch.testing.assert_close(actual, expected)


def test_loads_legacy_adaptive_checkpoint(tmp_path: Path) -> None:
    source = AdaptiveCNNStudent(in_features=768, out_dim=5, width=20, fc_hidden=128)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "config": {"dropout": 0.2},
        },
        checkpoint,
    )

    model, payload = load_checkpoint_model(checkpoint)

    assert isinstance(model, AdaptiveCNNStudent)
    assert model_input_features(model) == 768
    assert model_output_features(model) == 5
    assert resolve_windows(payload, None) == 39


def test_windows_override_wins_over_checkpoint_config() -> None:
    assert resolve_windows({"config": {"context_windows": 9}}, 4) == 4


def test_static_adaptive_max_pool_matches_pytorch_for_four_to_eight() -> None:
    x = torch.randn(3, 5, 4)
    expected = F.adaptive_max_pool1d(x, 8)
    actual = StaticAdaptiveMaxPool1d(input_size=4, output_size=8)(x)
    torch.testing.assert_close(actual, expected)


def test_prepared_four_channel_export_model_preserves_outputs() -> None:
    model = CNNMicroSequence(in_features=48, out_dim=5, dropout=0.2).eval()
    export_model = prepare_export_model(model, windows=9)
    assert isinstance(export_model.backbone.pool, StaticAdaptiveMaxPool1d)

    x = torch.randn(2, 9, 48)
    with torch.inference_mode():
        expected = model(x)
        actual = export_model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
