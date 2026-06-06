# H3 결과 — Feature Importance 기반 주차 후보 Rule (RandomForest 단독)

실행: `h3_rule/h3_rule.ipynb` · 모델 `RandomForestClassifier` 단독(no boosting) · `random_state=42`
생성일: 2026-06-06 · 표본 level 270–290, ≥10/12 접속 통제 · **H1 6mo·4피처·k=4 라벨(380명) 재실행**

> **해석 주의 (in-sample 운용점)**: 확정 주차 ground truth 없음. RF의 ROC-AUC/PR-AUC만 **5-fold OOF**(일반화 추정)이고, **최종 threshold sweep·rule 평가는 동일한 2,000명 H1 라벨에 대한 in-sample 운용점**(held-out 분할 없음)이다. 따라서 final_rule의 Precision/Recall/FPR은 "실제 주차 탐지 성능"이 아니라 **H1 후보 라벨을 보수적으로 근사하는 rule의 적합도**로 읽어야 한다. H1 라벨 자체가 후보(확정 아님)이므로 이중으로 후보 수준이다.

## 1. 표본 · 타깃

**H3 가설**: H1 후보군은 H1에서 사용하지 않은 관측 피처를 활용해 설명 가능한 단순 rule로 근사할 수 있을 것이다.

| 항목 | 값 |
|---|---|
| 학습 행수 (df_final) | 2,000 |
| 양성 `is_stagnant_cluster` | **380 (19.0%, 6mo·4피처·k=4)** 단일 target |

과거 현재성 보조 라벨(`is_current_parking_candidate` 52 · `is_high_confidence_candidate` 39)·`h1_current_candidates.csv`는 폐기(`temporal_external_validation` 삭제). **380 단일 라벨 대비 평가.**

## 2. 피처 풀 · 누수 가드(확장)

후보 13개 = Tier A raw state 7 + Tier B 비-H1 delta 3 + Tier C access 3.

- **Tier A (raw state)**: level · union_level · authentic_symbol_score · hexa_level_sum · log_exp · character_age_months · combat_power_latest(snapshots 마지막 유효월, winsor p01–p99; 270+ 표본 스케일 ~50M–350M)
- **Tier B (비-H1 delta)**: cumexp(log1p) avg + recent3/6 — **union·auth·cp·hexa delta 전면 제외**(= H1 4 채택축)
- **Tier C (access)**: access_active_months · access_ratio · access_recent
- **제외(누수)**: Δ{combat_power(+slog)·hexa·union_level·authentic_symbol} 전부 + 각 recent3/6 · hexa_fragments · (경계)Δlevel

**누수 가드 = H1 네 축(recent6 slog Δcp·Δhexa·log Δunion·log Δauth) 대비 `|corr|>0.85` drop → drop 0건.**
최대 절대상관 = `hexa_level_sum` 0.535(Δhexa축). raw state·cumexp는 H1 네 축과 무상관 수준 → 누수 아님(설계 전제 검증). 과거 2축(cp·hexa)만 보던 가드를 **4축으로 확장**(union·auth가 H1 채택축이 됨) 적용 후에도 drop 없음.

## 3. RandomForest 5-fold CV (out-of-fold)

| 지표 | 값 |
|---|---|
| ROC-AUC | **0.8495** |
| PR-AUC | 0.5365 |
| F1 (thr 0.5) | 0.4029 |
| Recall / Precision | 0.2921 / 0.6491 |

→ H1 4 채택축 **없이** raw-state + cumexp delta만으로 정체 군집을 ROC-AUC 0.85 근사(과거 471 라벨 0.853과 동급). 핵심 축 의도 제외 → 완전 복제 아님(상한 존재).

## 4. Feature Importance (permutation, n_repeats=20)

| 순위 | 피처 | perm | impurity |
|---:|---|---:|---:|
| 1 | **log1p_recent3_delta_cumexp** | 0.1079 | 0.160 |
| 2 | **combat_power_latest** | 0.1019 | 0.168 |
| 3 | **log1p_recent6_delta_cumexp** | 0.0976 | 0.125 |
| 4 | access_ratio | 0.0576 | 0.057 |
| 5 | hexa_level_sum | 0.0335 | 0.081 |
| 6 | log_exp | 0.0262 | 0.073 |
| 7 | log1p_avg_monthly_delta_cumexp | 0.0218 | 0.079 |
| 8 | level | 0.0210 | 0.050 |
| 9 | union_level | 0.0196 | 0.069 |
| 10 | authentic_symbol_score | 0.0154 | 0.059 |
| 11 | character_age_months | 0.0135 | 0.056 |
| 12 | access_active_months | **0.0018** | 0.017 |
| 13 | access_recent | **0.0009** | 0.008 |

- 최강 비-H1 판별자 = **recent3 누적EXP 증분(낮을수록 정체)** + **raw 전투력 절대값(낮을수록 정체)**. 둘 다 주차 운영정의(성장 동결 + 저재투자)와 직접 정합.
- **access family importance ≈ 0**(access_active_months 0.0018, access_recent 0.0009) — 표본 ≥10 접속 통제가 access 변별력을 죽임(CLAUDE.md 전제 정량 확인). access는 피처가 아니라 게이트로만 타당. (access_ratio만 0.058로 약한 잔여 신호.)

## 5. Rule 추출 → 운용점 선정

상위 2피처(cumexp recent3 · 전투력)로 depth-2 트리 → `log1p_recent3_delta_cumexp <= 30.84 AND combat_power_latest <= 127.7M` → 정체(loose). loose는 **over-flag(581>380)·FPR 0.199**. → `combat_power_latest` 임계 `T_cp`를 sweep해 **목표 FPR ≤ 0.10** 운용점 선정(cumexp 임계 0.30→ log 30.84 고정, access gate 고정).

### Sweep (cum<=30.84 & access_gate, vs is_stagnant_cluster 380)

| T_cp | flag | Precision | Recall | FPR |
|---:|---:|---:|---:|---:|
| 60M | 128 | 0.453 | 0.153 | 0.043 |
| **70M (채택)** | **220** | **0.500** | **0.289** | **0.068** |
| 80M | 314 | 0.471 | 0.389 | 0.102 |
| 90M | 388 | 0.479 | 0.489 | 0.125 |
| 100M | 447 | 0.468 | 0.550 | 0.147 |

T_cp=70M가 **FPR 0.068(≤0.10)**·Precision 0.500의 보수 임계라 채택. (80M부터 FPR 0.10 초과. FPR≤0.10 한도 내 최대 recall은 ~79M의 0.379.)

### 채택 운용 rule (`operating_point.json`)

```
정체(rule_op) IF  log1p_recent3_delta_cumexp <= 30.84
              AND combat_power_latest <= 70,000,000
최종(final_rule) IF  rule_op AND access_active_months >= 2
```

| rule | 양성수 |
|---|---:|
| stagnant_growth_rule (loose tree, 대조) | 581 |
| rule_op (운용점 성장정체) | 220 |
| access_gate 통과 | 2,000 |
| **final_rule** (rule_op AND gate) | **220** |

> `access_active_months >= 2` 게이트는 표본이 이미 ≥10 접속 통제라 **전원 통과**(in-sample no-op) → final_rule = rule_op. 공식 운용 시 dormant 유저 제외용 명시 요건으로 유지. (과거 471 run의 `valid_access_active_months` 게이트(1,804 통과)는 삭제된 `h1_current_candidates.csv` 출신 → 현 표본은 `access_active_months`로 대체.)

## 6. 평가표 (loose baseline vs 운용점 final_rule, vs is_stagnant_cluster 380, **in-sample**)

threshold sweep과 아래 표는 모두 학습에 쓴 동일 2,000명 H1 라벨에 대한 **in-sample 적합도**다(held-out 분할 없음). "탐지 성능"이 아니라 "H1 후보 라벨 근사도"로 해석한다.

| rule | 양성수 | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| stagnant_growth_rule (loose tree) | 581 | 0.446 | 0.682 | 0.539 | 0.199 |
| **final_rule (운용점)** | 220 | **0.500** | 0.289 | 0.367 | **0.068** |

→ 운용점이 헤드라인 FPR **0.199 → 0.068 (2.9배↓)**, precision 0.446→0.500. 목표 FPR ≤ 0.10 충족(in-sample 기준).

## 7. 해석 · H3 가설 판정

- **운용점 final_rule** = recent3 누적EXP 증분(log1p) ≤ 30.84 AND 전투력 ≤ 7,000만 AND 접속 ≥2 → **FPR(vs 380) 0.068**. 정밀↑·재현↓의 **보수 고정밀 운용점**.
- 비용: recall(vs 380) 0.289로 낮음 — over-flag 억제의 대가. 더 넓게 잡으려면 T_cp 80–100M(FPR 0.10–0.15, recall 0.39–0.55) 완화 가능(sweep 곡선 참조).
- **cumexp 주의**: 월별 누적EXP 보간 노이즈 있으나 recent3 증분 **방향**(근-제로 vs 양)은 견고 → 임계 rule엔 적합. 절대 magnitude 정밀도엔 의존 안 함.
- access 게이트는 ≥10 통제 표본이라 추가 변별 없음(전원 통과) — 보조 안전장치 성격.

**H3 판정 — 조작적 정의 기준 제한적 지지**: Feature Importance 파생 단순 rule(2조건+게이트)이 허용 FPR(**in-sample 0.068 ≤ 0.10**) 내에서 H1 성장 정체 후보를 설명 가능한 rule로 재현한다. 따라서 **H1 정체 후보 라벨 기준으로는 H3를 제한적으로 지지**한다. RF 5-fold OOF ROC-AUC 0.850도 raw-state+cumexp만으로 정체군집이 학습 가능함을 뒷받침한다.

해석 범위(실제 주차 ground truth로 확대 해석 금지): (a) sweep·rule 평가는 in-sample → "H1 라벨 근사도"이지 "실제 주차 탐지력"이 아님(실제 탐지력은 후속 외부검증 대상), (b) H1 라벨은 확정 아닌 proxy, (c) 보수 운용점이라 recall 낮음(목적 따라 T_cp 완화로 조정 가능).

## 산출물

| 파일 | 내용 |
|---|---|
| `data/h3_rule_eval.csv` | ocid별 stagnant_growth_rule/rule_op/access_gate/final_rule |
| `data/h3_metrics.csv` | 위 평가표 |
| `h3_rule/operating_point.json` | 채택 운용 rule(CUM_T·T_OP·식) |
| `h3_rule/figures/h3_importance.png` | permutation importance |
| `h3_rule/figures/h3_rule_tree.png` | depth-2 트리 |
| `h3_rule/figures/h3_threshold_sweep.png` | 운용점 sweep 곡선 |

재현: `jupyter nbconvert --to notebook --execute --inplace h3_rule/h3_rule.ipynb`
