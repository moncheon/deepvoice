# Run 0 audio data

`bootstrap-v1/audio/` contains locally generated PCM16 WAV fixtures and is intentionally ignored by Git. Its auditable ledger and summary remain trackable:

- `bootstrap-v1/manifest.jsonl`
- `bootstrap-v1/SUMMARY.json`
- `bootstrap-v1/splits.json`

Regenerate the binary audio only into a new, non-existing directory. The generator refuses overwrite so an experiment corpus cannot silently change in place.

```powershell
python run0/generate_audio.py bootstrap --output run0/data/bootstrap-v2
```

Bootstrap labels exercise the pipeline but are not real-world detector evidence. Performance training requires a reviewed source manifest and `generate_audio.py build`.

Production data uses three ignored binary locations:

- `staging-v1/`: downloaded-but-untrusted sources plus per-source `REVIEW.json` gates
- `sources-v1/`: the combined v2 source ledger and split-leakage audit
- `corpus-v1/`: 6-second component crops mixed into PCM16 files, exact splits, and `HASHES.sha256`

The generated-model dataset must also contain `MODEL_REVIEW.json` and
`WHEEL_REVIEW.json`; `corpus.py review-generated` writes the matching
`GENERATED_REVIEW.json` before it can be included. SONICS full songs remain a
challenge source and are not silently treated as isolated music stems.

The completed production artifact contains 24,000 WAV files (9,277,968,170
bytes, 181,193.5904375 seconds). Simultaneous mixtures are approximately six
seconds; sequential mixtures concatenate two six-second component crops plus a
bounded gap. `AUDIT.json` records zero failures, all eight label combinations
at 3,000 files each, and 8,764 parent-source keys confined to one split. Audio
is stored under `audio/C00` through `audio/C07`, keeping each directory below
the Hugging Face 10,000-files-per-directory limit.
