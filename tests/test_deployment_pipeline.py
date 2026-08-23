from __future__ import annotations

import numpy as np

from deployment.realtime_preprocess import RealtimePreprocessConfig, RealtimePreprocessor
from deployment.replay import replay_array


def test_replay_preserves_every_sample_in_order():
    source = np.arange(400, dtype=np.float32).reshape(100, 4)
    chunks = []

    stats = replay_array(
        source,
        sample_rate=2000.0,
        callback=lambda x: chunks.append(x.copy()),
        chunk_ms=7.0,
        realtime=False,
    )

    rebuilt = np.concatenate(chunks, axis=0)
    np.testing.assert_array_equal(rebuilt, source)
    assert stats["samples"] == 100
    assert stats["channels"] == 4


def test_realtime_preprocessing_is_chunk_boundary_invariant():
    rng = np.random.default_rng(7)
    # Enough signal for several 200 ms / 50%-overlap emissions.
    source = rng.normal(size=(1000, 4)).astype(np.float32)
    cfg = RealtimePreprocessConfig(
        sample_rate=2000.0,
        channels=4,
        window_size=0.2,
        overlap=0.5,
        context_windows=4,
    )

    one_shot = RealtimePreprocessor(cfg).push(source)

    streaming = RealtimePreprocessor(cfg)
    chunked = []
    for start in range(0, len(source), 37):
        chunked.extend(streaming.push(source[start : start + 37]))

    assert len(chunked) == len(one_shot)
    assert len(chunked) > 0
    for a, b in zip(chunked, one_shot):
        assert a.shape == (1, 4, 4 * 12)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
