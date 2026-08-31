# Run1 hybrid

Run1은 organizer Baseline의 PANNs presence 출력을 그대로 유지하고, 세 fake head에는 Baseline DF-Arena/HTDemucs 점수와 Run0 점수를 독립적으로 rank-fusion하는 제출 후보입니다.

## 고정 검증 설계

- generator-disjoint 8 folds: voice 6, music 2
- exact codec LOGO 5 folds
- fold별 고정 panel 256개, 별도 ECDF calibration 512개
- epoch 1–6과 fake head별 Baseline weight를 nested LOGO로 선택하고, robust ADS 기준 epoch 5를 최종 채택
- gate: Baseline 대비 paired-bootstrap robust ADS 개선이 `0.003` 이상
- presence는 Baseline PANNs로 고정하여 CPS 회귀를 방어
- 파일 간 통계 사용 금지: 제출 시 ECDF knot와 weight는 calibration에서 미리 고정

`reports/LOGO_PLAN.json`은 추적 가능한 compact 설계 요약입니다. 실제 sample ID 계획과 split은 private HF dataset의 immutable plan 경로에 보관합니다.

## 안전·재현성

- 외부 wheel은 다운로드 후 파일 검토를 통과해야 설치됩니다.
- corpus shard, 원본 오디오, backbone, Baseline weight는 크기와 SHA-256을 확인한 뒤 사용합니다.
- 학습 checkpoint는 epoch마다 tensor-safe safetensors와 JSON state로 저장하며 manifest를 마지막에 원자적으로 기록합니다.
- Baseline 캐시는 모델 branch와 개별 오디오 SHA-256으로만 keyed 됩니다.
- HF GPU 사용은 `tools/hf_budget.py`의 `$10` hard cap 사전 예약·사후 정산을 통과해야 합니다.
- 제출 builder는 LOGO gate, ZIP root, 전체 artifact hash, 9 GB archive/30 GB expanded 제한을 확인합니다.
- 최종 후보는 L4에서 실제 제출 ZIP으로 1,200파일을 실행하고 내부 50분 gate를 통과해야 제출할 수 있습니다.

## 주요 명령

```powershell
python -m unittest discover -s tests -v

python tools\assemble_logo_ledger.py `
  --plan work\logo_plan.json `
  --baseline work\baseline_predictions.csv `
  --fold-root work\logo_evidence `
  --source-bundle-sha256 79dfa911c903e697675f03af1e4d9e6808a1bd6f54855b953ac7f03d2fa19b34 `
  --output work\logo_ledger.csv

python tools\select_hybrid.py `
  --ledger work\logo_ledger.csv `
  --output reports\LOGO_REPORT.json `
  --bootstrap-iterations 1000
```

최종 학습, fusion spec 생성, 제출 ZIP 제작은 `LOGO_REPORT.json`의 `status`가 `passed`일 때만 진행합니다. 현재 13-fold generator·codec LOGO는 epoch 5에서 gate를 통과했고, 실제 제출 ZIP은 NVIDIA L4에서 1,200개 파일을 1,138.5초에 처리해 내부 50분 gate를 통과했습니다.
