from __future__ import annotations

import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from deepvoice_run0.audio import decode_audio, to_mid_side
from deepvoice_run0.segmentation import segment_audio, select_segment_starts


class AudioSegmentationTests(unittest.TestCase):
    def test_mid_side_mono_and_stereo(self) -> None:
        mono = np.asarray([[1.0, -1.0]], dtype=np.float32)
        mono_ms = to_mid_side(mono)
        np.testing.assert_allclose(mono_ms[0], mono[0])
        np.testing.assert_allclose(mono_ms[1], 0.0)
        stereo = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(to_mid_side(stereo), [[0.5, 0.5], [0.5, -0.5]])

    def test_short_audio_uses_reflect_padding_and_mask(self) -> None:
        audio = np.vstack((np.linspace(-1, 1, 100, dtype=np.float32), np.zeros(100, dtype=np.float32)))
        result = segment_audio(
            audio,
            sample_rate=100,
            window_seconds=4.0,
            hop_seconds=2.0,
            all_windows_until_seconds=16.0,
            uniform_segments=12,
            novelty_segments=4,
            max_segments=16,
        )
        self.assertEqual(result.samples.shape, (1, 2, 400))
        self.assertEqual(int(result.valid_mask.sum()), 100)

    def test_long_audio_is_bounded(self) -> None:
        rng = np.random.default_rng(2608)
        audio = rng.normal(size=60 * 100).astype(np.float32)
        starts = select_segment_starts(
            audio,
            sample_rate=100,
            window_seconds=4.0,
            hop_seconds=2.0,
            all_windows_until_seconds=16.0,
            uniform_segments=12,
            novelty_segments=4,
            max_segments=16,
        )
        self.assertLessEqual(len(starts), 16)
        self.assertGreaterEqual(len(starts), 12)
        self.assertEqual(starts, sorted(set(starts)))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is unavailable")
    def test_ffmpeg_decode_preserves_stereo_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                frames = [struct.pack("<hh", 1000, -1000) for _ in range(800)]
                handle.writeframes(b"".join(frames))
            decoded = decode_audio(path, 16_000)
            self.assertEqual(decoded.samples.shape[0], 2)
            self.assertEqual(decoded.original_sample_rate, 8000)
            self.assertEqual(decoded.original_channels, 2)
            self.assertGreater(decoded.samples.shape[1], 1500)

    def test_native_pcm_fast_path_preserves_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.wav"
            expected = np.asarray([0, 1024, -2048, 32767], dtype=np.int16)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(expected.astype("<i2").tobytes())
            decoded = decode_audio(path, 16_000)
            self.assertEqual(decoded.codec_name, "pcm_s16le")
            np.testing.assert_allclose(decoded.samples[0], expected.astype(np.float32) / 32768.0)


if __name__ == "__main__":
    unittest.main()
