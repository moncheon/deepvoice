# deepvoice

## deepvoice

이 저장소는 DACON의 [딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회](https://www.dacon.io/competitions/official/236749/overview/description)에 제출한 오프라인 오디오 탐지 코드와 학습·검증 기록을 관리합니다.

- **목적:** 신종 보이스피싱, 음성 사칭, 허위정보 생성에 악용되는 AI 생성·변조 오디오를 탐지합니다.
- **판정 대상:** 한 파일에 음성, 음악 또는 두 성분이 동시에·순차적으로 포함될 수 있으며, 각 성분의 존재 여부와 AI 생성 여부를 판정합니다.
- **출력:** 파일마다 `FILE_FAKE_PROB`, `VOICE_FAKE_PROB`, `MUSIC_FAKE_PROB`, `VOICE_PRESENT_PROB`, `MUSIC_PRESENT_PROB`의 다섯 확률을 생성합니다.
- **Real/Fake 기준:** AI가 새로 생성한 음성·음악 성분은 Fake(1), 실제 원천 성분은 Real(0)입니다. 음성이나 음악 중 하나라도 Fake이면 파일 전체를 Fake로 판정합니다. 품질 개선·잡음 제거·음량 조정처럼 성분 자체를 새로 만들지 않는 후처리는 Real로 봅니다.
- **주최:** 행정안전부, 한국지능정보사회진흥원(NIA)
- **주관:** 국립과학수사연구원
- **운영:** 데이콘(DACON)

공식 [평가식](https://www.dacon.io/competitions/official/236749/overview/evaluation)은 다음과 같습니다.

```text
Score = 0.9 × ADS + 0.1 × CPS
ADS = 0.5 × (1 - File EER) + 0.2 × (1 - Voice EER) + 0.3 × (1 - Music EER)
CPS = 0.5 × Voice Presence ROC-AUC + 0.5 × Music Presence ROC-AUC
```

제출물은 `model/`, `script.py`, `requirements.txt`만 포함하는 `submit.zip` 코드 제출 형식이며, 1,200개 평가 파일을 인터넷이 차단된 NVIDIA L4 환경에서 60분 이내에 추론해야 합니다.

## 제출

2026-08-30 리더보드 스냅샷 기준 결과입니다. 표는 organizer Baseline, GPU 학습 Run0 순서입니다.

| 제출 | 기술 특징 | 총점 | ADS | CPS | 공개 순위 |
| --- | --- | ---: | ---: | ---: | ---: |
| **Baseline** | PANNs로 음성·음악 존재 확률 추정 → HTDemucs 성분 분리 → DF-Arena 1B로 성분별 Fake 판정 → segment 최댓값 집계 | `0.69091` | `0.65775` | `0.989323` | 104위 |
| **Run0 v1** | WavLM Base+ 음성 branch + spectral residual music branch, mid/side 입력, 최대 16개 segment, gated MIL, 5개 multi-task head와 OOF fusion; 공개·직접 생성 corpus로 A100 학습 | `0.6517645476` | `0.6183571429` | `0.9524311905` | 144위 |

Run0 submission ID는 `76229`, DACON 실행 시간은 `1분 46초`입니다. Baseline 대비 총점은 `-0.0391454524`이며, 공식 가중치로 분해하면 격차의 약 90.6%가 ADS에서 발생합니다. 따라서 다음 라운드는 unseen generator·codec에 대한 ADS 일반화를 우선하고, CPS는 Baseline의 전용 presence branch와 Run0을 결합하는 hybrid를 먼저 검증합니다.

대용량 organizer weight는 GitHub 이력에 넣지 않고 private Hugging Face 저장소에 보관합니다. 고정 링크와 SHA-256은 [`modellink.yaml`](modellink.yaml), 설명과 복구 절차는 [`format/baseline_submit/model/HF_STORAGE.md`](format/baseline_submit/model/HF_STORAGE.md)에 있습니다.

