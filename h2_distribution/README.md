# H2: H1 후보군 분포 검정

## 목적

H2는 H1 클러스터링으로 도출한 성장 정체/주차 후보군이 특정 레벨 구간 또는 직업 계열에 불균일하게 분포하는지 검정한다.

## 입력

- `data/features_monthly.csv`
- `data/cluster_labels.csv`
- 보조 분석: `data/h1_current_candidates.csv`

기본 라벨:

- `is_stagnant_cluster`: H1 K-Means 후보 cluster, 412명
- `is_current_parking_candidate`: 현재성 조건까지 적용한 보조 후보, 63명
- `is_high_confidence_candidate`: H1 후보와 현재성 후보의 교집합, 35명

## 현재 결과

- 기본 분석 표본: 1,979명
- H1 클러스터 후보: 412명, 20.82%
- 보조 현재성 후보: 63명, 3.34%
- 보조 고신뢰 현재성 후보: 35명, 1.86%

| 검정 | p | Holm 보정 p | Cramer's V | 판정 |
|---|---:|---:|---:|---|
| 레벨 구간 x H1 후보 | 1.486e-61 | 2.972e-61 | 0.376 | 유의 |
| 직업 계열 x H1 후보 | 0.7476 | 0.7476 | 0.031 | 유의하지 않음 |

H1 후보는 `270-279` 구간에서 40.14%로 강하게 집중되고, `286-290` 구간에서는 2.82%로 낮다. 직업 계열 집중은 확인되지 않는다.

63명 현재성 후보와 35명 고신뢰 후보는 보조 분석으로 유지하며, 둘 다 레벨 구간 방향성은 같고 직업 계열 집중은 유의하지 않다.

상세 수치는 `RESULTS.md`와 `results.json`에 기록되어 있다.

## 실행

```bash
python h2_distribution/run_analysis.py
jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb
```
