"""Shared constants for training, inference, and submission validation."""

PREDICTION_COLUMNS = (
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)

MODEL_OUTPUT_KEYS = (
    "file_fake",
    "voice_fake",
    "music_fake",
    "voice_present",
    "music_present",
)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}

OFFICIAL_WEIGHTS = {
    "file_fake": 0.45,
    "voice_fake": 0.18,
    "music_fake": 0.27,
    "voice_present": 0.05,
    "music_present": 0.05,
}
