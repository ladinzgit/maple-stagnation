# H3: 설명 가능한 규칙 근사

**상태: 완료** (2026-06-06, H1 6mo·k=4 라벨 380명 기준)

## 가설

H1 후보군은 H1에서 사용하지 않은 관측 피처를 활용해 설명 가능한 단순 규칙으로 근사할 수 있을 것이다.

## 설계 원칙

- 모델: **RandomForestClassifier 단독** (boosting 제외)
- 입력 피처: H1이 쓴 4축(cp/hexa/union/auth delta) **전면 제외**, raw state 7개 + 비-H1 cumexp delta 3개 + access 3개 = 13피처
- 누수 가드: H1 네 축과 |corr| > 0.85인 피처 drop → **실제 drop 0건** (최대 0.535)
- 평가 대상: `is_stagnant_cluster` (380명) 단일 라벨, in-sample (held-out 분할 없음)

## 결과

### RandomForest 5-fold OOF

| 지표 | 값 |
|---|---:|
| ROC-AUC | **0.8495** |
| PR-AUC | 0.5365 |

### Feature Importance (permutation 상위 3)

| 순위 | 피처 | 중요도 |
|---:|---|---:|
| 1 | `log1p_recent3_delta_cumexp` (낮을수록 정체) | 0.108 |
| 2 | `combat_power_latest` (낮을수록 정체) | 0.102 |
| 3 | `log1p_recent6_delta_cumexp` | 0.098 |

access family (access_active_months: 0.0018, access_recent: 0.0009) — ≥10 접속 통제로 변별력 없음. access는 피처가 아닌 게이트.

### 채택 운영 규칙

```
정체(stagnant) IF  log1p_recent3_delta_cumexp <= 30.84
                AND combat_power_latest <= 70,000,000
                AND access_active_months >= 2
```

| rule | 양성수 | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| loose tree (대조) | 581 | 0.446 | 0.682 | 0.199 |
| **최종 운영점** | **220** | **0.500** | **0.289** | **0.068** |

전투력 임계 70M은 FPR ≤ 0.10 조건 내 보수 운영점. T_cp 80~100M으로 완화 시 recall 0.39~0.55, FPR 0.10~0.15.

**판정: H3 제한적 지지** — H1 채택 피처 없이 raw-state + cumexp delta만으로 후보군을 허용 FPR (in-sample 0.068 ≤ 0.10) 내에서 재현.

> 규칙 평가는 held-out 외부 검증이 아닌 동일 표본 in-sample 적합도입니다. H1 라벨 자체가 proxy이므로 실제 주차 탐지력으로 확대 해석하지 않습니다.

## 실행

```bash
jupyter nbconvert --to notebook --execute --inplace h3_rule/h3_rule.ipynb
```

## 산출물

| 파일 | 내용 |
|---|---|
| `operating_point.json` | 채택 운영 규칙 임계값 |
| `data/h3_rule_eval.csv` | ocid별 예측 결과 |
| `data/h3_metrics.csv` | 평가 지표 |
| `figures/` | 중요도, depth-2 트리, sweep 곡선 그림 |
| `RESULT.md` | 상세 결과 |
