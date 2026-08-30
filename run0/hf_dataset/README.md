---
license: other
pretty_name: Deepvoice Run 0 private generated audio
task_categories:
- audio-classification
language:
- ko
- en
tags:
- audio
- deepfake-detection
- synthetic-speech
- synthetic-music
---

# Deepvoice Run 0 private generated audio

This private, non-commercial research dataset contains generic, non-identity-
cloned outputs produced for defensive deepfake-detector training.

The source ledger records every WAV SHA-256, generic prompt ID, random seed,
generator family, model repository, and immutable model revision. It also
contains:

- `MODEL_REVIEW.json`: safe-file, license, revision, and model-file hashes;
- `WHEEL_REVIEW.json`: exact Python wheel hashes and native-member inventory;
- `SUMMARY.json`: output counts, bytes, and source-manifest hash.

Licensing is composite and must be evaluated per record:

- `facebook/mms-tts-kor`: CC BY-NC 4.0;
- `facebook/musicgen-small`: CC BY-NC 4.0;
- `cvssp/audioldm2-music`: CC BY-NC-SA 4.0.

The prompts are project-authored and do not request imitation of a named
person or artist. The audio and manifests are not detector-performance
evidence by themselves and must remain private.

`generated-v1` at immutable dataset revision
`192e75d6f8eb1f9af3a56750ff84b6adfce92a6b` contains 1,320 reviewed WAV
files (355,254,496 audio bytes): 1,000 Korean MMS-TTS voice files, 256
MusicGen instrumental files, and 64 AudioLDM2 instrumental files. Its source
manifest SHA-256 is
`24c127d08c5fce2d00ddf9cdd79706837e40d026ec8797f2b56631ebe8eb2504`.

`corpus-v1` contains 24,000 audited PCM16 WAV files derived from 11,771
reviewed sources. Every one of the eight voice/music real/fake/presence label
combinations has 3,000 files; train/dev/holdout counts are
16,800/3,600/3,600. The corpus is 9,277,968,170 bytes and its manifest SHA-256
is `37372ad3f6dc5e37494960a309808fef3a413b10928f94143034de0ddb5c1423`.
`AUDIT.json` reports zero format/hash/split/fixture failures.
The audio is partitioned into `audio/C00` through `audio/C07` with 3,000 files
per directory to remain below the Hub per-directory file limit.
