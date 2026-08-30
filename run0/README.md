# Deepvoice Run 0

`pre-plan/0-analysis.md`와 `pre-plan/1-strategy.md`를 실행 코드로 옮긴 첫 foundation입니다. 목표는 baseline의 PANNs → HTDemucs → 1B detector 두 번 → max 집계를, 원본 mixture 중심의 compact multi-task detector와 누수 방지 데이터 ledger로 교체하는 것입니다.

## 현재 완료 범위

- ffmpeg 단일 decode, 16 kHz 변환, stereo mid/side view
- 4초/2초 창과 긴 파일의 12 uniform + 최대 4 novelty segment
- SHA-256·license·consent·generator 계보를 요구하는 JSONL manifest
- source group 고정 split과 generator-family holdout
- 동시/순차 voice/music 혼합 및 label 자동 도출
- 절차 생성 bootstrap WAV와 허가 원본 기반 corpus builder
- source manifest v2: 서로 다른 dataset을 split별 bucket으로 결합하고 동일 음원 hash까지 atomic split
- pin 고정 다운로드 → `REVIEW.json` 재해시 → 실행의 3단계 corpus/model gate
- 공개 real voice/music 수집기와 비사칭 MMS-TTS/MusicGen/AudioLDM2 생성 작업
- 검토된 source 11,771개에서 만든 실제 학습 WAV 24,000개와 전체 hash audit
- compact local SSL voice branch + multi-resolution spectral/residual music branch
- gated MIL, 다섯 multi-task head, 공식 가중 masked loss, pairwise ranking
- dependency-light EER/AUC/공식 weighted score와 OOF-only logistic fusion
- offline Dacon 제출 진입점과 schema/ID/finite-range 검사
- private Hugging Face Docker Space용 bounded worker scaffold

## 아직 증명되지 않은 범위

검토된 실제 corpus와 생성 오디오 구축은 완료했지만, 아직 학습된 `run0.safetensors`의 LOGO/worst-group 점수는 없습니다. 따라서 생성·데이터 파이프라인 통과를 detector 성능·L4 처리시간·VRAM 기준 통과로 해석하지 않습니다. 실제 채택은 `pre-plan/1-strategy.md`의 독립 gate를 통과해야 합니다.

## 로컬 검증

```powershell
python -m unittest discover -s run0/tests -t run0 -v
python run0/tools/manifest_cli.py validate run0/examples/manifest.example.jsonl
python run0/generate_audio.py --help
python run0/script.py --help
python run0/train.py --help
python run0/evaluate.py --help
python run0/benchmark.py --help
```

현재 경량 환경에서 30개 unit test와 CLI parsing이 동작합니다. 학습/추론에는 `torch`, `transformers`, `safetensors`, 로컬 backbone/weights가 필요합니다. 모델 hub network loading은 꺼져 있습니다.

## 데이터 준비

### 즉시 생성되는 bootstrap 오디오

```powershell
python run0/generate_audio.py bootstrap
python run0/tools/manifest_cli.py validate run0/data/bootstrap-v1/manifest.jsonl
python run0/tools/manifest_cli.py audit-files run0/data/bootstrap-v1/manifest.jsonl
python run0/tools/manifest_cli.py split run0/data/bootstrap-v1/manifest.jsonl --output run0/data/bootstrap-v1/splits.json --stratify-labels
```

이 명령은 8개 voice/music Real-like/Fake-like 조합, 동시·순차 혼합, 5개 채널 profile의 PCM16 WAV를 생성합니다. 사람 음성이나 곡을 사용하지 않는 DSP fixture이므로 개인정보 문제 없이 전체 학습 경로를 smoke-test할 수 있습니다. 다만 Real-like도 실제 인간 녹음이 아니므로 이 자료의 score는 현실 탐지 성능 증거가 아닙니다. 기본 학습기는 이를 거부하며 smoke test일 때만 `--allow-fixture-smoke-training`으로 명시적으로 허용합니다.

현재 `bootstrap-v1`에는 실제 생성된 WAV 32개(12,297,104 bytes, 총 238.863125초)가 있으며, train/dev/holdout은 label-stratified 16/8/8입니다. 자세한 hash와 분포는 [SUMMARY.json](data/bootstrap-v1/SUMMARY.json)에 기록했습니다. WAV binary는 Git에서 제외하지만 manifest, summary, split은 추적 가능합니다.

### 실제 학습 corpus v2

1. source의 실제 라이선스와 consent 근거를 검토합니다.
2. split 전에 각 원본/파생본을 동일한 `split_group`으로 기록합니다.
3. `manifest_cli.py audit-files`로 파일 존재와 SHA-256을 확인합니다.
4. generator family를 숨긴 최종 holdout은 별도 ledger로 동결합니다.
5. train 실행 시에만 `--acknowledge-licenses-reviewed`를 명시합니다.

소스 카탈로그는 [corpus_sources.json](corpus_sources.json)에 URL/revision/license를 고정합니다. `fetch`는 저장만 하고, `review`가 위험 파일·라이선스 문서·전체 SHA-256을 기록한 뒤에야 `prepare`가 읽습니다. v2 builder는 dataset마다 네 bucket을 모두 요구하지 않고, 누수 없이 미리 배정된 split 안에서 이질적인 voice/music source를 pooling합니다. 동일 identity/content/parent/audio SHA-256은 항상 같은 split에 남습니다.

```powershell
python run0/corpus.py fetch mlaad-tiny zeroth-korean-hf fma-instrumental-hf sonics
python run0/corpus.py review mlaad-tiny zeroth-korean-hf fma-instrumental-hf sonics
python run0/corpus.py review-generated run0/data/staging-v1/generated-open-v1/generated-v1/sources.jsonl
python run0/corpus.py prepare --generated-manifest run0/data/staging-v1/generated-open-v1/generated-v1/sources.jsonl
python run0/corpus.py build --per-combination 3000 --duration 6
python run0/corpus.py audit
python run0/train.py --manifest run0/data/corpus-v1/manifest.jsonl --splits run0/data/corpus-v1/splits.json --acknowledge-licenses-reviewed
```

현재 실제 artifact는 다음과 같습니다.

- source ledger: 11,771개, 12개 component/label/split bucket 모두 존재, leakage 0
- 직접 생성: MMS-TTS-KOR 1,000개, MusicGen 256개, AudioLDM2 64개
- production corpus: 24,000개, 8개 label 조합별 3,000개
- 저장 layout: `audio/C00`~`audio/C07`에 각 3,000개(HF의 디렉터리당 10,000파일 제한 준수)
- split: train 16,800 / dev 3,600 / holdout 3,600
- 총량: 9,277,968,170 bytes, 181,193.5904375초
- corpus manifest SHA-256: `37372ad3f6dc5e37494960a309808fef3a413b10928f94143034de0ddb5c1423`
- hash audit: 24,000개 실패 0, parent source 8,764개 split 누수 0
- private dataset: `datahj/deepvoice-run0-corpus-v1` revision `4fbc8bef72d8a0b0d3d9767ded5b9f80b2938d0c`

`--duration 6`은 component crop 길이입니다. 동시 혼합은 약 6초이고, 순차 혼합은 두 component와 최대 0.4초 gap을 이어 붙여 약 12초가 됩니다.

SONICS의 full song은 challenge 보관용입니다. 보컬/반주 stem이 검토되기 전에는 music-only source로 가져오지 않아 presence label 오염을 막습니다. 실제 음악은 FMA의 `Vocal content: none`과 적응 가능한 per-track Creative Commons attribution이 동시에 확인된 행만 사용합니다.

외부 checkpoint는 먼저 저장한 뒤 source, revision, license, hash를 검토하고 `run0/model/backbone/`에 복사합니다. 다운로드와 loader 실행을 한 명령으로 연결하지 않습니다.

## 제출 추론

필수 artifact 계약은 [model/MODEL_README.md](model/MODEL_README.md)에 있습니다.

```powershell
python run0/script.py `
  --test-dir data/test `
  --sample-submission data/sample_submission.csv `
  --output output/submission.csv `
  --device cuda
```

기본은 strict입니다. `--fail-soft`는 파일별 실패를 `errors.csv`에 기록하고 그 파일에만 0.5를 쓰는 명시적 비상 모드입니다. 평가 파일 간 통계, pseudo-label, cross-file normalization은 사용하지 않습니다.

OOF/holdout CSV는 `evaluate.py`로 전체 및 generator/source/codec group별 평가할 수 있습니다. 실제 L4에서는 `benchmark.py --enforce-gates`가 1,200개 50분 추정과 peak VRAM 20 GiB 내부 gate를 검사합니다.

## 제출 패키지

학습 artifact가 준비된 뒤 다음 스크립트가 허용 목록만 ZIP으로 만듭니다.

```powershell
powershell -ExecutionPolicy Bypass -File run0/tools/build_submission.ps1
```

스크립트는 `script.py`, `config.toml`, `requirements.txt`, `deepvoice_run0/`, `model/`만 포함하며 weights/backbone이 없으면 실패합니다.

Run0 학습 가중치가 생기기 전 리더보드 파이프라인 자체를 점검할 때는 제공 baseline을 control로 패키징합니다.

```powershell
powershell -ExecutionPolicy Bypass -File run0/tools/build_baseline_control_submission.ps1
```

기본 출력은 `run0/build/submit-baseline-control.zip`입니다. 빌더는 포함된 세 checkpoint의 SHA-256을 먼저 검증하고, Zip64 store 방식으로 최상위 `model/`, `script.py`, `requirements.txt`만 생성한 뒤 10 GB/32 GB 제한과 경로 구조를 다시 검사합니다. 옆의 `submit-baseline-control.manifest.json`에는 최종 ZIP 해시와 검증 결과가 기록됩니다. 이 제출은 Run0 성능이 아니라 organizer baseline의 기준점만 측정합니다.

## Hugging Face 실행

[hf_space/DEPLOY.md](hf_space/DEPLOY.md)는 private Docker Space와 일회성 HF Job 실행 절차입니다. private worker/corpus/model 저장소를 생성했고 Space commit `e4a885d3d06093c41f0e967a535f3a61b9d4e18b`의 build와 인증된 `/health`를 확인한 뒤 pause했습니다. 하드웨어 변경 권한이 없는 fine-grained token에서는 Space GPU API가 401이므로, 고정 digest의 PyTorch image와 private Space source를 read-only mount한 4시간 제한 Job을 사용했습니다. 완료 Job `6a93901e984507d9db4ec523`은 L40S에서 2,666초 실행됐고, wheel 구조/hash 검토 → 모델 safetensors/hash/license 검토 → 3-file CUDA smoke → 1,320개 생성 → private dataset upload를 순서대로 통과했습니다. 실제 L40S 비용은 약 $1.33이며, 앞선 검토 단계에서 차단된 41초 Job까지 포함하면 약 $1.35입니다.
