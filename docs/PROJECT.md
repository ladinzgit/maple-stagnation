# 프로젝트 현황

## 연구 목적

Nexon OpenAPI로 직접 관측 가능한 성장 이력과 접속 행동을 이용해 메이플스토리 주차 후보군을 탐색한다. 공개 API에서는 주간 보스 수행 기록이나 메소 생산량을 직접 조회할 수 없으므로, 본 프로젝트의 결과는 확정 주차 유저 라벨이 아니라 운영 검증용 후보 라벨이다.

## 현재 데이터 기준

- 수집 대상: 270~290레벨 본캐 후보
- active 기준: 12개월(2025-06~2026-05) 스냅샷 중 **≥10개월 접속**(`MIN_CHECKPOINT_MONTHS=10`) — 접속은 표본 통제변인
- 본캐 표본: 2,000명
- 피처 표본: 2,000명
- H1 클러스터링 유효 표본: 2,000명
- H1 분석 윈도우: **최근 6개월(2025-12~2026-05)** delta = `recent6_delta_*` 컬럼 (주차=현재 행동; features_monthly 단독 재현)
- H2/H3: Done (380 6mo 라벨 재실행 2026-06-06)

## 연구 가설

| 구분 | 가설 문장 | 분석 방법 |
|---|---|---|
| H1 | 최근 6개월간 주요 성장 지표가 낮은 활성 캐릭터들은 일반 성장 캐릭터들과 구분되는 성장 정체 군집으로 식별될 것이다. | K-Means 클러스터링, DBSCAN 보조 검증 |
| H2 | H1에서 도출한 성장 정체 후보군의 비율은 캐릭터 속성인 레벨 구간 및 직업 계열에 따라 불균일하게 분포할 것이다. | 카이제곱 독립성 검정 |
| H3 | H1 후보군은 H1에서 사용하지 않은 관측 피처를 활용해 설명 가능한 단순 rule로 근사할 수 있을 것이다. | RandomForest 중요도, 얕은 Decision Tree, threshold sweep |

## H1 결과 (6mo 전환 2026-06-05)

H1은 성장 정체 특성이 강한 활성 캐릭터들이 일반 성장 캐릭터와 구분되는 군집으로 식별되는지 확인한다. `feature_selection.ipynb`에서 최근 6개월 delta(`recent6_delta_*` 컬럼)로 4-시스템 family-diverse 조건으로 피처를 선정했다.

최종 선택 피처 (4-시스템 다축):

| Alias | 실제 값 | 시스템 |
|---|---|---|
| `cp_slog` | 전투력 월 Δ signed-log | 전투력 |
| `hexa_avg` | `avg_monthly_delta_hexa` clip≥0 | 헥사 |
| `union_log` | `log1p(Δunion_level)` | 유니온 |
| `auth_log` | `log1p(Δauthentic_symbol)` | 어센틱 심볼 |

선택 결과:

| 항목 | 값 |
|---|---:|
| K-Means k | 4 (elbow; best-sil k=6) |
| Silhouette @k4 | 0.357 |
| 주차 후보 cluster id | 1 |
| 최종 H1 주차 후보 | **380명 (19.0%)** |
| park(stag≥4) enrich / recall | 4.37x / 83.0% |

클러스터별 인원:

| Cluster | 인원 | 해석 |
|---|---:|---|
| 0 | 882명 | 일반 성장 군집 |
| 1 | 380명 | 성장 정체/주차 후보 (전투력 감소 + 4-시스템 저재투자) |
| 2 | 551명 | 고성장 군집 (헥사 특화) |
| 3 | 187명 | 전투력만 증가, 유니온·심볼 정체 |

DBSCAN(3군집)이 정체 380 ↔ cluster1(355) 강한 정합. 상세 피처 프로파일은 `h1_clustering/RESULT.md`에 기록했다.

## H2 결과 (380, 2026-06-06)

`data/cluster_labels.csv`의 H1 클러스터 후보 라벨(380) × 레벨 구간/직업 계열 분포 검정(Chi-Square, α=0.05). 단일 H2 가설을 두 속성 축으로 검정한다.

- **레벨 구간**: 유의(χ²=38.54, p=4.27e-9, Holm 8.54e-9, Cramer's V 0.139). 표준화 잔차 기준 `280-285` 집중(+3.18, 비율 23.04%, OR 1.96), `286-290` 희박(-4.47, 비율 11.72%). 270-279는 비율 높으나 n=107로 비유의.
- **직업 계열**: 무유의(χ²=6.89, p=0.142, Cramer's V 0.059). 실질적 집중 없음.
- 결론: **H2 부분 지지**(조작적 정의 = H1 정체 후보 라벨 기준). 레벨 구간에서는 280-285 집중이 확인되어 지지되지만, 직업 계열에서는 불균일 분포가 확인되지 않았다. 검정 대상이 후보 라벨이므로 실제 주차 분포 일반화는 라벨 타당성 전제.
- **일반화 주의**: 표본 = 270-290 본캐 + ≥10/12 접속 + 전투력 ~50M↑ active/capable(휴면·저레벨 없음) → 표본 설계 내에서만 일반화. 레벨 표본 불균형(107/1176/717)이라 저레벨(270-279) 해석 보수적. 상세: `h2_distribution/RESULTS.md`.

## H3 결과 (380, 2026-06-06)

H3는 H1 성장 정체 후보군을 H1에서 사용하지 않은 관측 피처로 근사하고 접속 게이트를 결합해 설명 가능한 단순 운영 rule을 만든다(RandomForest 단독, 13피처).

- 학습/평가 target: `is_stagnant_cluster` = **380명(6mo)** 단일 (현재성 라벨 폐기 — `temporal_external_validation` 삭제)
- 누수 가드(H1 네 축) 적용 → union·auth·cp·hexa delta 제외, |corr|>0.85 점검 drop 0(최대 0.535).
- RF 5-fold ROC-AUC **0.850**. importance 1·2·3위 = recent3 cumexp Δ · combat_power_latest · recent6 cumexp Δ. access ≈ 0.
- 운용 rule = `log1p_recent3_delta_cumexp≤30.84 AND combat_power_latest≤70M AND access_active_months≥2` → Precision 0.500 / Recall 0.289 / **FPR 0.068**(≤0.10) → **조작적 정의 기준 제한적 지지**(H1 정체 후보 라벨 기준; ROC-AUC 0.850 OOF 뒷받침).
- **해석 범위**: sweep·rule 평가는 in-sample(held-out 없음) → "H1 후보 라벨 근사도"(실제 주차 탐지력은 후속 외부검증 대상). H1 라벨=proxy.
- 주의: `is_stagnant_cluster` 자체는 확정 주차 유저 라벨이 아니라 성장 정체 후보 라벨이다(이중 후보). 외부 검증 미수행. 상세: `h3_rule/RESULT.md`.

## 주요 산출물

| 파일 | 역할 |
|---|---|
| `data/main_characters.csv` | active checkpoint 조건을 통과한 본캐 2,000명 |
| `data/features_monthly.csv` | 12개월 월별 성장/접속 피처 |
| `data/hexa_fragments.csv` | HEXA 조각 소비 이력 |
| `data/cluster_labels.csv` | H1 cluster label 및 주차 후보 cluster 여부 (380명) — H2/H3 단일 입력 |
| `h1_clustering/RESULT.md` | H1 최종 결과와 cluster별 특성 |
| `h2_distribution/RESULTS.md` | H2 분포 검정 결과 |
