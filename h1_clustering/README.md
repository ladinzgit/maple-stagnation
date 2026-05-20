# 가설 1 — K-Means / DBSCAN 클러스터링

주차 유저가 일반 유저와 구별되는 군집을 형성하는지 검증한다.

## 입력

`df_final` (1,463행) — `eda/eda.ipynb` Sec 10에서 정의된 전처리 완료 데이터

## 피처 세트

| 세트 | 피처 |
|---|---|
| A (권장) | `avg_monthly_delta_level`, `avg_monthly_delta_combat_power`, `avg_monthly_delta_union_level`, `avg_monthly_delta_authentic_symbol`, `arcane_stagnant` |
| B (비교용) | 원본 5개 delta 컬럼 (`arcane_stagnant` 대신 `avg_monthly_delta_arcane_symbol` 사용) |

## 방법

1. `StandardScaler` 정규화
2. K-Means — Elbow Method + Silhouette Score로 최적 k 선정
3. DBSCAN — 보조 검증
4. 두 세트 Silhouette Score 비교 → 최종 클러스터 레이블 결정

## 출력

- 최적 클러스터 레이블 (H2/H3에서 재사용)
- 클러스터별 delta 분포 시각화
