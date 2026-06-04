# 프로젝트 현황

## 연구 목적

Nexon OpenAPI로 직접 관측 가능한 성장 이력과 접속 행동을 이용해 메이플스토리 주차 후보군을 탐색한다. 공개 API에서는 주간 보스 수행 기록이나 메소 생산량을 직접 조회할 수 없으므로, 본 프로젝트의 결과는 확정 주차 유저 라벨이 아니라 운영 검증용 후보 라벨이다.

## 현재 데이터 기준

- 수집 대상: 270~290레벨 본캐 후보
- active 기준: `2025-06`, `2025-12`, `2026-05` 세 checkpoint month 모두에서 접속 관측
- 기준 선정 이유:
  - 6월, 12월은 메이플스토리 대형 이벤트가 주로 진행되는 시점
  - `2026-05`는 현재 실험 기준에서 실제 활동 중인 유저를 대상으로 하기 위한 시점
- 본캐 표본: 2,000명
- 피처 표본: 2,000명
- H1 클러스터링 유효 표본: 1,979명
- H2 기본 검정 표본: 1,979명
- H2 현재성 보조 검정 표본: 1,884명

## H1 결과

`feature_selection.ipynb`에서 clipping/winsorize 정책을 적용한 뒤 family-diverse 조건으로 피처 조합을 다시 탐색했다.

최종 선택 피처:

| Alias | 실제 값 |
|---|---|
| `cumexp_avg` | `log1p_avg_monthly_delta_cumexp` |
| `union_avg` | `avg_monthly_delta_union_level`, `[0, p99]` clipping |
| `access_ratio` | 월별 접속 비율 |

선택 결과:

| 항목 | 값 |
|---|---:|
| K-Means k | 3 |
| Silhouette | 0.5446 |
| 주차 후보 cluster id | 1 |
| 최종 H1 주차 후보 | 412명 |

클러스터별 인원:

| Cluster | 인원 | 해석 |
|---|---:|---|
| 0 | 1,419명 | 일반 성장 군집 |
| 1 | 412명 | 성장 정체/주차 후보 군집 |
| 2 | 148명 | 유니온 성장 특화 군집 |

상세 피처 프로파일은 `h1_clustering/RESULT.md`에 기록했다.

## H2 결과

`data/cluster_labels.csv`의 H1 클러스터 후보 라벨을 기준으로 사전 정의한 분포 검정을 수행했다. `data/h1_current_candidates.csv`의 현재성 라벨은 보조 분석으로 유지했다.

- H1 클러스터 후보: 412명, 20.82%
- 보조 현재성 후보: 63명, 3.34%
- 보조 고신뢰 후보: 35명, 1.86%

| 검정 | chi-square | p | Holm 보정 p | Cramer's V | 판정 |
|---|---:|---:|---:|---:|---|
| 레벨 구간 x H1 후보 | 280.123 | 1.486e-61 | 2.972e-61 | 0.376 | 유의 |
| 직업 계열 x H1 후보 | 1.936 | 0.7476 | 0.7476 | 0.031 | 유의하지 않음 |

H1 후보는 `270-279` 구간에서 40.14%로 강하게 집중되고, `286-290` 구간에서는 2.82%로 낮다. 직업 계열 집중은 확인되지 않는다. 상세 결과는 `h2_distribution/RESULTS.md`에 기록했다.

## H3 설계

H3는 아직 구현 예정 단계다. 목표는 H1의 성장 정체 cluster를 다른 관측 피처로 근사하고, 현재성 접속 게이트를 결합해 운영 가능한 rule을 만드는 것이다.

- 1단계 target: `data/cluster_labels.csv`의 `is_stagnant_cluster` = 1, 412명
- 최종 rule 평가 target: `data/h1_current_candidates.csv`의 `is_current_parking_candidate`
- 민감도 평가 target: `is_high_confidence_candidate`
- 주의: `is_stagnant_cluster` 자체는 확정 주차 유저 라벨이 아니라 성장 정체 후보 라벨이다.

## 주요 산출물

| 파일 | 역할 |
|---|---|
| `data/main_characters.csv` | active checkpoint 조건을 통과한 본캐 2,000명 |
| `data/features_monthly.csv` | 12개월 월별 성장/접속 피처 |
| `data/hexa_fragments.csv` | HEXA 조각 소비 이력 |
| `data/cluster_labels.csv` | H1 cluster label 및 주차 후보 cluster 여부 |
| `data/h1_current_candidates.csv` | 현재성 조건이 반영된 후보 라벨 |
| `h1_clustering/RESULT.md` | H1 최종 결과와 cluster별 특성 |
| `h2_distribution/RESULTS.md` | H2 분포 검정 결과 |
