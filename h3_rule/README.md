# H3: Feature Importance 기반 Rule 평가

## 상태

**구현·실행 완료(2026-06-06 재실행)** — `h3_rule.ipynb`, RandomForestClassifier 단독, raw-state + 비-H1 delta 피처.

> H1 6mo 전환(4피처·k=4) target `is_stagnant_cluster` **380명(19.0%)** 단일 라벨로 재실행 완료. union·auth가 H1 채택축이 되어 H3 피처풀에서 제외했고, 누수 가드를 **H1 네 축**으로 확장 적용(drop 0). 결과는 `RESULT.md` 참조.

## 목적

H3 가설은 **H1 후보군은 H1에서 사용하지 않은 관측 피처를 활용해 설명 가능한 단순 rule로 근사할 수 있을 것이다**이다. H3는 H1의 성장 정체 후보 cluster를 다른 관측 피처로 근사하고, 현재성 접속 조건을 결합해 운영 가능한 rule을 만드는 실험이다. 분류기 성능 자체가 목적이 아니라, feature importance를 통해 단순하고 설명 가능한 rule 후보를 도출하는 것이 목적이다.

## 입력 라벨

| 라벨 | 파일 | 현재 인원 | 용도 |
|---|---|---:|---|
| `is_stagnant_cluster` | `data/cluster_labels.csv` | **380명 (6mo)** | 학습 + 최종 rule 평가 단일 target |

(과거 `is_current_parking_candidate`(52)·`is_high_confidence_candidate`(39) 라벨은 폐기 — `temporal_external_validation` 삭제. 최종 rule은 `is_stagnant_cluster`(380) 대비 평가.)

## 설계 원칙

- 모델은 **RandomForestClassifier 단독**으로 한다. Boosting(XGBoost 등)은 사용하지 않는다.
- H1은 성장 **delta** 축 **4개(`cp_slog`=전투력 Δ, `hexa_avg`=헥사 Δ, `union_log`=유니온 Δ, `auth_log`=어센틱 심볼 Δ)**를 썼다(6mo). H3는 이와 **다른 신호인 raw state값**(현재 스냅샷 절대값)을 신규 입력으로 추가한다: `level`, `union_level`, `authentic_symbol_score`, `hexa_level_sum`, `log_exp`, `character_age_months`, `combat_power_latest`. raw state는 delta가 아니므로 H1 경계의 직접 복제가 아니다.
  - `combat_power_latest`는 `features_monthly.csv`에 없다 → `monthly_snapshots_raw.csv`에서 ocid별 마지막 유효월 `combat_power`를 추출해 머지한다.
- **6mo·4축 반영(적용 완료)**: 과거엔 union·authentic delta를 비-cp/hexa Tier B 피처로 넣었으나, **union·auth가 H1 채택축이므로 H3 입력에서 제외**했다. 남는 비-H1 delta = `log1p_avg_monthly_delta_cumexp`(+recent3/6). 6mo window 기준 delta·raw state로 재실행 완료.
- **leakage 가드**: **H1 네 축(slog Δcp, Δhexa, log Δunion, log Δauth)**을 inline 재계산해 각 후보 피처와의 `|corr|`를 점검하고, `|corr| > 0.85`면 입력에서 drop한다.
- **제외(누수)**: `avg_monthly_delta_{combat_power(+slog),hexa,union_level,authentic_symbol}` 및 각 `recent3/6_delta_*`, hexa_fragments family (= H1 4-시스템 채택축 전부 + 파생). 경계 `avg_monthly_delta_level`·recent level delta는 기본 제외, 민감도 옵션으로만 둔다.
- **RandomForest 단조불변 주의**: RF split은 monotonic 변환에 불변이라 같은 변수의 raw와 log를 동시에 넣으면 트리는 동일하고 importance만 두 컬럼으로 희석된다. 변수당 한 형태만 쓰고, log는 rule threshold 가독성이나 비율형 파생피처용으로만 쓴다.
- 접속 여부는 수집 단계의 사전 필터로 추가 제거하지 않는다. importance 산출에는 포함하되(표본이 ≥10 통제라 변별력은 제한적), 최종 rule에서만 별도 현재성 게이트로 쓴다.

## 예정 방법

1. `features_monthly` + `cluster_labels`(ocid 머지) + `monthly_snapshots_raw` 로드, `combat_power_latest` 파생 머지
2. 전처리 inline(union_level NaN→0, raw cp/exp winsor p99, 피처 명시 조립, `y=is_stagnant_cluster`)
3. 누수 가드(H1 **네 축**[recent6 slog Δcp·Δhexa·log Δunion·log Δauth] 재계산 → corr 점검 → drop)
4. RandomForestClassifier(`class_weight='balanced'`, `random_state=42`), 5-fold StratifiedKFold CV → ROC-AUC/PR-AUC/F1
5. permutation importance(주) + impurity importance(교차검증)
6. 상위 2~3개 피처로 얕은 DecisionTree(max_depth 2~3) → `export_text`로 명시 임계값 rule 추출
7. threshold sweep로 Precision/Recall/FPR 곡선
8. 최종 rule = 성장 정체 rule AND `access_active_months >= 2` (삭제된 `h1_current_candidates.csv`의 `valid_access_active_months` 대체; ≥10 통제 표본이라 전원 통과)
9. 단일 라벨(**380 cluster, 6mo**) 대비 Precision/Recall/F1/FPR/ROC-AUC 보고 (현재성 라벨 폐기)

## 해석 주의

본 프로젝트에는 확정 주차 유저 ground truth가 없다. 따라서 H3의 모든 지표는 H1/H2에서 만든 후보 라벨 대비 성능이며, 실제 주차 유저 검증 성능으로 해석하지 않는다.
