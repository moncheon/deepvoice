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

첫 A100 완주 run은 엄격한 미관측-generator holdout에서 weighted score 0.5161,
voice-fake EER 0.7698로 내부 gate를 통과하지 못했습니다. 이 checkpoint는 private
`runs/` 증거로만 보존하고 승격·제출하지 않습니다. 이는 소스 분리 진단 결과이지
리더보드 성능은 아닙니다. 다음 production run은 speaker/content/parent group 원자성을
유지하면서 MLAAD·MMS·MusicGen·AudioLDM2의 coarse domain을 train/dev/holdout 모두에
포함하는 별도 split을 사용합니다. 그 run도 holdout gate와 L4 benchmark를 통과하기
전에는 제출 가능한 성능으로 간주하지 않습니다.

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

원래 `corpus-v1/splits.json`은 MMS-TTS-KOR와 AudioLDM2를 완전히 감춘
generator-holdout 진단용입니다. production 후보는 다음 명령으로 별도 생성합니다.
두 split을 같은 이름으로 덮어쓰지 않습니다.

```powershell
python run0/tools/manifest_cli.py split run0/data/corpus-v1/manifest.jsonl `
  --output run0/build/domain-balanced-splits-v1/splits.json `
  --summary run0/build/domain-balanced-splits-v1/SUMMARY.json `
  --seed 2609 --train-ratio 0.7 --dev-ratio 0.15 --stratify-domains
```

현재 생성 결과는 train 16,761 / dev 3,551 / holdout 3,688, split-group 누수 0이며,
세 split 모두 MLAAD·MMS·MusicGen·AudioLDM2 표본을 포함합니다. assignment SHA-256은
`e7a0d6caec337007f1f5a752dda0eefa66eb135417fe88523e9d44acb79c97d7`입니다.

`--duration 6`은 component crop 길이입니다. 동시 혼합은 약 6초이고, 순차 혼합은 두 component와 최대 0.4초 gap을 이어 붙여 약 12초가 됩니다.

SONICS의 full song은 challenge 보관용입니다. 보컬/반주 stem이 검토되기 전에는 music-only source로 가져오지 않아 presence label 오염을 막습니다. 실제 음악은 FMA의 `Vocal content: none`과 적응 가능한 per-track Creative Commons attribution이 동시에 확인된 행만 사용합니다.

외부 checkpoint는 먼저 저장한 뒤 source, revision, license, hash를 검토하고 `run0/model/backbone/`에 복사합니다. 다운로드와 loader 실행을 한 명령으로 연결하지 않습니다.

Run0 v1 backbone은 `microsoft/wavlm-base-plus` revision `4c66d4806a428f2e922ccfa1a962776e232d487b`로 고정했습니다. [backbone_source.json](backbone_source.json)은 원본 checkpoint를 실행하기 전에 확인한 파일 allowlist, SHA-256, CC BY-SA 3.0 라이선스 원문 revision을 기록합니다. GPU Job은 이 ledger와 private source snapshot이 정확히 일치한 뒤에만 원본을 열고 tensor-safe `model.safetensors`로 변환합니다.

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

스크립트는 최상위에 `script.py`, `requirements.txt`, `model/`만 두며, 설정과 Python 패키지는 `model/` 안에 넣습니다. 성능·분산·라이선스 gate를 통과한 weights/backbone이 없으면 실패합니다.
대형 safetensors를 메모리에 모으지 않도록 low-memory Zip64 store writer로 파일을
하나씩 추가합니다. 최종 ZIP의 루트 구조·10GB/32GB 제한·SHA-256은 별도로 다시
검증합니다.

Run0 학습 가중치가 생기기 전 리더보드 파이프라인 자체를 점검할 때는 제공 baseline을 control로 패키징합니다.

```powershell
powershell -ExecutionPolicy Bypass -File run0/tools/build_baseline_control_submission.ps1
```

기본 출력은 `run0/build/submit-baseline-control.zip`입니다. 빌더는 포함된 세 checkpoint의 SHA-256을 먼저 검증하고, Zip64 store 방식으로 최상위 `model/`, `script.py`, `requirements.txt`만 생성한 뒤 10 GB/32 GB 제한과 경로 구조를 다시 검사합니다. 옆의 `submit-baseline-control.manifest.json`에는 최종 ZIP 해시와 검증 결과가 기록됩니다. 이 제출은 Run0 성능이 아니라 organizer baseline의 기준점만 측정합니다.

## Hugging Face 실행

[hf_space/DEPLOY.md](hf_space/DEPLOY.md)는 private Docker Space와 일회성 HF Job 실행 절차입니다. private worker/corpus/model 저장소를 생성했고 Space commit `e4a885d3d06093c41f0e967a535f3a61b9d4e18b`의 build와 인증된 `/health`를 확인한 뒤 pause했습니다. 하드웨어 변경 권한이 없는 fine-grained token에서는 Space GPU API가 401이므로, 고정 digest의 PyTorch image와 private Space source를 read-only mount한 4시간 제한 Job을 사용했습니다. 완료 Job `6a93901e984507d9db4ec523`은 L40S에서 2,666초 실행됐고, wheel 구조/hash 검토 → 모델 safetensors/hash/license 검토 → 3-file CUDA smoke → 1,320개 생성 → private dataset upload를 순서대로 통과했습니다. 실제 L40S 비용은 약 $1.33이며, 앞선 검토 단계에서 차단된 41초 Job까지 포함하면 약 $1.35입니다.

성능 학습 Job은 `train_job_entrypoint.sh`를 사용합니다. 8개 corpus shard와 WavLM source snapshot을 각각 private dataset/model volume에서 읽고 모든 해시를 재검증한 뒤 A100 batch-2 smoke, 유효 batch 8, epoch당 4,800개 가중표본, 최대 10 epoch 학습을 수행합니다. 매 epoch full dev 공식 점수로 best checkpoint를 선택하고, 선택이 끝난 뒤 holdout을 한 번만 평가합니다. PyTorch 2.6.0 CUDA 12.4 공식 이미지는 digest `sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee`로 고정합니다.

production split은 Job 환경변수 `RUN0_SPLIT_RELATIVE=splits/domain-balanced-v1.json`으로
명시합니다. 학습 sampler도 exact generator 모델명별 역빈도 대신 label + coarse
generator domain별 역빈도를 사용하여 희귀 모델명 하나가 epoch을 과점하지 않게 합니다.

L4 benchmark는 Hub mount의 safetensors를 직접 mmap하지 않습니다. 필요한 모델 파일을
Job-local `/tmp`로 복사하면서 artifact ledger와 다시 대조한 뒤 로드하고, 16/12/8
segment 측정이 하나 끝날 때마다 private model repo의
`reports/L4_BENCHMARK_PROGRESS.json`에 중간 결과를 저장합니다.

산출물은 먼저 private model repo의 `runs/<run-id>/`에만 업로드됩니다. 아래 도구는 immutable repository revision에서 파일을 내려받아 `ARTIFACTS.json`의 크기와 SHA-256, 성능 gate, run/source provenance를 확인한 뒤에만 로컬 `model/`과 private `final/run0-v1/`로 승격합니다.

```powershell
python run0/tools/promote_training_run.py `
  --repo-id datahj/deepvoice-run0-model-v1 `
  --revision <40-character-model-repo-commit> `
  --run-id <run-id> `
  --source-commit <40-character-source-commit>
```

## DACON Run0 v1 실측 결과

2026-08-30 20:47 KST 기준으로 GPU 학습 Run0 v1을 DACON 대회
`236749`에 제출하고 채점 및 공개 리더보드 반영을 확인했습니다.

- submission ID: `76229`
- 제출 파일: `submit-run0-v1-67255dc.zip`
- 제출 ZIP SHA-256: `18144818efbed1753ac094f9df912f972bd266b64731ea8511fa5554e7d9370e`
- 제출 시각: `2026-08-30 20:45:13 KST`
- 총점: `0.6517645476`
- ADS: `0.6183571429`
- CPS: `0.9524311905`
- DACON 실행 시간: `1분 46초`
- 공개 리더보드: 팀 `데봅`, `144위`, 표시 총점 `0.65176`

이 값은 로컬 holdout 점수가 아니라 DACON의 실제 코드 제출 채점 결과입니다.
반대로 공개 리더보드 점수만으로 private/final leaderboard 성능을 보장하지는 않습니다.

### Baseline 대비 memo

동일 리더보드 스냅샷에서 DACON.GM Baseline은 104위, Run0 v1은 144위였습니다.

| 지표 | DACON.GM Baseline | Run0 v1 | Run0 차이 |
| --- | ---: | ---: | ---: |
| 총점 | `0.69091` | `0.6517645476` | `-0.0391454524` |
| ADS | `0.65775` | `0.6183571429` | `-0.0393928571` |
| CPS | `0.989323` | `0.9524311905` | `-0.0368918095` |

공식 총점은 `0.9 * ADS + 0.1 * CPS`입니다. 상세 ADS/CPS 값으로 분해하면
총점 격차의 약 90.6%가 ADS에서 발생합니다. CPS만 Baseline 수준으로 맞출 때의
예상 총점은 `0.6554537286`인 반면, ADS만 맞추면 `0.6872181191`입니다.
따라서 다음 라운드는 ADS의 unseen generator/codec 일반화를 우선하고, CPS는
Baseline의 전용 presence branch를 Run0과 결합하는 hybrid를 먼저 검증합니다.
