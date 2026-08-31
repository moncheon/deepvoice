from __future__ import annotations

import unittest

from deepvoice_hybrid.logo import atomic_generators, audit_logo_plan, build_logo_plan


def _row(index: int, generator: str | None, codec: str, group: str, kind: str) -> dict[str, object]:
    voice_present = int(kind in {"real_voice", "fake_voice", "both"})
    music_present = int(kind in {"real_music", "fake_music", "both"})
    voice_fake = 1 if kind in {"fake_voice", "both"} else 0 if voice_present else None
    music_fake = 1 if kind in {"fake_music", "both"} else 0 if music_present else None
    return {
        "sample_id": f"sample-{index:05d}",
        "split_group": group,
        "generator_model": generator,
        "codec_chain": codec,
        "file_fake": int(bool(voice_fake) or bool(music_fake)),
        "voice_present": voice_present,
        "music_present": music_present,
        "voice_fake": voice_fake,
        "music_fake": music_fake,
    }


class LogoTests(unittest.TestCase):
    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        codecs = [f"codec-{index}" for index in range(5)]
        voice = [f"voice-{index}" for index in range(8)]
        music = ["facebook/musicgen-small", "cvssp/audioldm2-music"]
        index = 0
        for repeat in range(40):
            for codec_index, codec in enumerate(codecs):
                rows.append(_row(index, None, codec, f"real-v-{index}", "real_voice")); index += 1
                rows.append(_row(index, None, codec, f"real-m-{index}", "real_music")); index += 1
                generator = voice[(repeat + codec_index) % len(voice)]
                rows.append(_row(index, generator, codec, f"fake-v-{index}", "fake_voice")); index += 1
                music_generator = music[(repeat + codec_index) % len(music)]
                rows.append(_row(index, music_generator, codec, f"fake-m-{index}", "fake_music")); index += 1
        return rows

    def test_atomic_generator_parsing(self) -> None:
        row = {"generator_model": "Voice-X+facebook/musicgen-small"}
        voice, music = atomic_generators(row)
        self.assertEqual(voice, ("voice-x",))
        self.assertEqual(music, ("facebook/musicgen-small",))

    def test_plan_has_generator_and_exact_codec_folds_without_leakage(self) -> None:
        rows = self._rows()
        plan = build_logo_plan(
            rows,
            seed=19,
            calibration_size=20,
            panel_size=12,
            voice_folds=2,
            music_folds=2,
        )
        self.assertEqual(len(plan["folds"]), 9)
        self.assertEqual(sum(fold["axis"] == "generator" for fold in plan["folds"]), 4)
        self.assertEqual(sum(fold["axis"] == "codec" for fold in plan["folds"]), 5)
        audit_logo_plan(rows, plan)


if __name__ == "__main__":
    unittest.main()
